use std::str::FromStr;

use regex::Regex;

use crate::{extract::{Extractor, Gap, Quartet, ReadMode, ReleaseInfo, Ripper, TrackExtractor}, integrity::IntegrityChecker, toc::{Toc, TocEntry, TocRaw}, track::{TestAndCopy, TrackEntry, TrackError, TrackErrorData, TrackErrorRange}, translate::{Translator, TranslatorCombined}, util::Time};
use simple_text_decode::DecodedText;

use super::{split_tracks, Parser, ParsedLog, ParserCombined, ParsedLogCombined, ParserTrack};

lazy_static! {
    static ref RIPPER_VERSION: Regex = Regex::new(r"EZ CD Audio Converter (\S+)").unwrap();
    static ref RELEASE_INFO: Regex = Regex::new(r"(?m)^Log creation date:.*\n\s*(?P<relinfo>.+?)\s*\n\s*Device: ").unwrap();
    static ref USED_DRIVE: Regex = Regex::new(r"(?m)^Device:\s*(?:\[[^\]]*\])?\s*(?P<drive>.+?)\s*$").unwrap();
    // Neither the mount point nor the trailing firmware is part of the AccurateRip name
    static ref DRIVE_TRIM: Regex = Regex::new(r"\s+\S{1,4}$").unwrap();

    static ref PROFILE: Regex = Regex::new(r"(?m)^Profile\s*: (?P<value>.+?)\s*$").unwrap();
    static ref READ_MODE: Regex = Regex::new(r"(?m)^Read mode\s*: (?P<value>.+?)\s*$").unwrap();
    static ref USE_C2: Regex = Regex::new(r"(?m)^C2 pointers\s*: (?P<boolean>Enabled|Disabled)").unwrap();
    static ref ACCURATE_STREAM: Regex = Regex::new(r"(?m)^AccurateCDDA\s*: (?P<boolean>Enabled|Disabled)").unwrap();
    static ref DRIVE_READ_CACHE: Regex = Regex::new(r"(?m)^Drive read cache\s*: (?P<boolean>Enabled|Disabled)").unwrap();
    static ref VERIFY_AUDIO: Regex = Regex::new(r"(?m)^Verify audio\s*: (?P<boolean>Enabled|Disabled)").unwrap();
    static ref READ_OFFSET: Regex = Regex::new(r"(?m)^Sample offset\s*: (?P<value>[+-]?\d+)").unwrap();
    static ref DETECT_PREGAPS: Regex = Regex::new(r"(?m)^Detect pre-gaps:?\s*: (?P<boolean>Enabled|Disabled)").unwrap();
    static ref CALCULATE_REPLAYGAIN: Regex = Regex::new(r"(?m)^Calculate (?:EBU R 128 \| )?ReplayGain\s*: (?P<boolean>Enabled|Disabled)").unwrap();
    static ref NORMALIZE: Regex = Regex::new(r"(?m)^(?:Apply ReplayGain|Loudness Normalization|Peak Normalization|Gain)\s*: (?P<value>.+?)\s*$").unwrap();
    static ref DELETE_SILENCE: Regex = Regex::new(r"(?m)^Delete (?:leading|trailing) digital silence\s*: (?P<boolean>Enabled|Disabled)").unwrap();
    static ref OUTPUT_FORMAT: Regex = Regex::new(r"(?m)^Output format\s*: \.?(?P<value>\w+)").unwrap();
    static ref VERIFY_COLUMN: Regex = Regex::new(r"(?m)^Track +Status +CRC(?P<verify> \(copy\) +CRC \(test\)| +CRC verify)?").unwrap();

    static ref TOC: Regex = Regex::new(r"(?m)^ *\[[X ]\] +(?P<track>\d+) +(?:\d+:\d{2}\.\d{2} +\d+:\d{2}\.\d{2} +)?(?P<start_sector>\d+) +(?P<end_sector>\d+)(?P<rest>.*)$").unwrap();
    static ref PREGAP: Regex = Regex::new(r"^ +(?P<time>\d+:\d{2}\.\d{2})").unwrap();
    static ref PREEMPHASIS: Regex = Regex::new(r" (?P<value>No|Yes|De-emphasized)\b").unwrap();

    static ref DETAILS: Regex = Regex::new(r"(?m)^Details\s*\n-+").unwrap();
    static ref SPLIT_TRACKS: Regex = Regex::new(r"(?m)^Track (\d+)\s*$").unwrap();
    static ref FILENAME: Regex = Regex::new(r#"(?m)^\s+Filename "(?P<value>.*)"\s*$"#).unwrap();
    static ref FINISHED: Regex = Regex::new(r"(?m)^\s+Finished (?:successfully|with errors)\s*$").unwrap();
    static ref TEST_CRC: Regex = Regex::new(r"(?m)^\s+CRC (?:\(test\)|verify)\s*: (?P<value>[0-9A-F]{8})\s*$").unwrap();
    static ref COPY_CRC: Regex = Regex::new(r"(?m)^\s+CRC(?: \(copy\))?\s*: (?P<value>[0-9A-F]{8})\s*$").unwrap();
    static ref SECTOR_ERROR: Regex = Regex::new(r"(?m)^\s+(?:(?P<kind>Read|Data|Synchronization) error at (?P<at>\d+:\d{2}:\d{2})|(?P<pos>\d+:\d{2}:\d{2})(?P<kinds>(?: +(?:Read|Data|Synchronization|CRC) error)+))\s*$").unwrap();
    static ref ERROR_KIND: Regex = Regex::new(r"(Read|Data|Synchronization|CRC) error").unwrap();
}

pub struct EzcdParser {
    encoded_log: DecodedText,
}

struct EzcdParserSingle {
    log: String,
    language: String,
}

struct EzcdParserTrack {
    num: u8,
    is_range: bool,
    pregap: Option<Time>,
    preemphasis: Option<bool>,
    raw: String,
}

impl EzcdParser {
    pub fn new(encoded_log: DecodedText) -> EzcdParser {
        EzcdParser {
            encoded_log,
        }
    }
}

impl EzcdParserSingle {
    pub fn new(log: String) -> EzcdParserSingle {
        let (language, _) = EzcdParserSingle::translate(log.clone());
        EzcdParserSingle {
            log,
            language,
        }
    }

    fn boolean_matcher(&self, regex: &Regex) -> Quartet {
        let captures = regex.captures(&self.log);
        match captures {
            Some(captures) => {
                let value = captures.name("boolean").unwrap().as_str();
                match value {
                    "Enabled" => Quartet::True,
                    "Disabled" => Quartet::False,
                    _ => Quartet::Unknown,
                }
            },
            None => Quartet::Unknown,
        }
    }

    fn string_match(&self, regex: &Regex) -> String {
        match regex.captures(&self.log) {
            Some(val) => val.name("value").unwrap().as_str().trim().to_string(),
            None => String::default(),
        }
    }

    fn optional_match<T>(&self, regex: &Regex) -> Option<T>
    where
        T: FromStr, {
        regex.captures(&self.log).and_then(|val| {
            val.name("value").and_then(|v| v.as_str().trim().parse::<T>().ok())
        })
    }

    fn toc_extras(&self) -> Vec<(u8, Option<Time>, Option<bool>)> {
        TOC.captures_iter(&self.log).filter_map(|captures| {
            let num = captures["track"].parse::<u8>().ok()?;
            let rest = &captures["rest"];
            let pregap = PREGAP.captures(rest).map(|pregap| Time::from_mm_ss_ff(&pregap["time"]));
            let preemphasis = PREEMPHASIS.captures(rest).map(|preemphasis| &preemphasis["value"] != "No");
            Some((num, pregap, preemphasis))
        }).collect()
    }

    fn tracks(&self) -> Vec<EzcdParserTrack> {
        let details = DETAILS.find(&self.log).map_or(self.log.as_str(), |m| &self.log[m.end()..]);
        let is_range = self.string_match(&PROFILE) == "Range";
        let toc_extras = self.toc_extras();

        split_tracks(details, &SPLIT_TRACKS).into_iter().map(|(num, raw)| {
            let extras = toc_extras.iter().find(|(track, _, _)| *track == num);
            EzcdParserTrack::new(
                num,
                is_range,
                extras.and_then(|(_, pregap, _)| *pregap),
                extras.and_then(|(_, _, preemphasis)| *preemphasis),
                raw.to_owned(),
            )
        }).collect()
    }
}

impl ParserCombined for EzcdParser {
    fn parse_combined(&self) -> ParsedLogCombined {
        let parsed_logs: Vec<ParsedLog> = vec![EzcdParserSingle::new(self.encoded_log.text.trim().to_string()).parse()];

        ParsedLogCombined {
            parsed_logs,
            encoding: self.encoded_log.orig_encoding.to_string()
        }
    }
}

impl TranslatorCombined for EzcdParser {
    fn translate_combined(&self) -> String {
        self.encoded_log.text.clone()
    }
}

impl Parser for EzcdParserSingle {}

impl Extractor for EzcdParserSingle {
    fn extract_ripper(&self) -> Ripper {
        Ripper::EZCD
    }

    fn extract_ripper_version(&self) -> String {
        let captures = RIPPER_VERSION.captures(&self.log);
        match captures {
            Some(captures) => captures.get(1).unwrap().as_str().to_string(),
            None => String::from("Unknown"),
        }
    }

    fn extract_language(&self) -> String {
        self.language.clone()
    }

    fn extract_release_info(&self) -> ReleaseInfo {
        let captures = RELEASE_INFO.captures(&self.log);
        match captures {
            Some(captures) => {
                let relinfo = captures.name("relinfo").unwrap().as_str();
                match relinfo.split_once(" - ") {
                    Some((artist, title)) => ReleaseInfo::new(artist.trim().to_owned(), title.trim().to_owned()),
                    None => ReleaseInfo::new(String::new(), relinfo.trim().to_owned()),
                }
            },
            None => ReleaseInfo::default(),
        }
    }

    fn extract_drive(&self) -> String {
        let captures = USED_DRIVE.captures(&self.log);
        match captures {
            Some(captures) => DRIVE_TRIM.replace(captures["drive"].trim(), "").trim().to_string(),
            None => String::default(),
        }
    }

    fn extract_read_offset(&self) -> Option<i16> {
        self.optional_match::<i16>(&READ_OFFSET)
    }

    fn extract_accurate_stream(&self) -> Quartet {
        self.boolean_matcher(&ACCURATE_STREAM)
    }

    fn extract_defeat_audio_cache(&self) -> Quartet {
        match self.boolean_matcher(&DRIVE_READ_CACHE) {
            Quartet::True => Quartet::False,
            Quartet::False => Quartet::True,
            other => other,
        }
    }

    fn extract_use_c2(&self) -> Quartet {
        self.boolean_matcher(&USE_C2)
    }

    fn extract_delete_silence(&self) -> Quartet {
        let deletes: Vec<bool> = DELETE_SILENCE.captures_iter(&self.log).map(|captures| &captures["boolean"] == "Enabled").collect();
        if deletes.contains(&true) {
            Quartet::True
        } else if deletes.is_empty() {
            Quartet::Unknown
        } else {
            Quartet::False
        }
    }

    fn extract_use_null_samples(&self) -> Quartet {
        Quartet::True
    }

    fn extract_test_and_copy(&self) -> Quartet {
        match self.boolean_matcher(&VERIFY_AUDIO) {
            Quartet::Unknown => match VERIFY_COLUMN.captures(&self.log) {
                Some(captures) => if captures.name("verify").is_some() { Quartet::True } else { Quartet::False },
                None => Quartet::Unknown,
            },
            other => other,
        }
    }

    // Image profiles log no gain lines, but gain that was never calculated was never applied
    fn extract_normalize(&self) -> Quartet {
        let gains: Vec<String> = NORMALIZE.captures_iter(&self.log).map(|captures| captures["value"].to_owned()).collect();
        if gains.iter().any(|gain| gain != "Disabled") {
            Quartet::True
        } else if !gains.is_empty() {
            Quartet::False
        } else {
            match self.boolean_matcher(&CALCULATE_REPLAYGAIN) {
                Quartet::False => Quartet::False,
                _ => Quartet::Unknown,
            }
        }
    }

    fn extract_read_mode(&self) -> ReadMode {
        match self.string_match(&READ_MODE).as_str() {
            "Normal" | "Error detection" | "Error detection (compatibility)" | "Simple error detection" | "Advanced error detection" => ReadMode::Burst,
            _ => ReadMode::Unknown,
        }
    }

    fn extract_gap_handling(&self) -> Gap {
        match self.boolean_matcher(&DETECT_PREGAPS) {
            Quartet::True => Gap::AppendNoHtoa,
            Quartet::False => Gap::AppendUndetected,
            _ => Gap::Unknown,
        }
    }

    fn extract_audio_encoder(&self) -> Vec<String> {
        let captures = OUTPUT_FORMAT.captures(&self.log);
        match captures {
            Some(captures) => vec![captures["value"].to_ascii_lowercase()],
            None => {
                let mut extensions: Vec<String> = self.tracks().iter().filter_map(|track| {
                    std::path::Path::new(&track.string_match(&FILENAME))
                        .extension()
                        .map(|extension| extension.to_string_lossy().to_ascii_lowercase())
                }).collect();
                extensions.sort();
                extensions.dedup();
                extensions
            },
        }
    }

    fn extract_toc(&self) -> Toc {
        let entries: Vec<TocEntry> = TOC.captures_iter(&self.log).filter_map(|captures| {
            let start_sector = captures["start_sector"].parse::<u32>().ok()?;
            let end_sector = captures["end_sector"].parse::<u32>().ok()?;
            Some(TocEntry::new(
                captures["track"].parse::<u32>().ok()?,
                Time::from_frames(start_sector),
                Time::from_frames((end_sector + 1).checked_sub(start_sector)?),
                start_sector,
                end_sector,
            ))
        }).collect();

        Toc::new(TocRaw::new(entries))
    }

    fn extract_tracks(&self) -> Vec<TrackEntry> {
        self.tracks().iter().map(|track| track.parse_track()).collect()
    }
}

impl Translator for EzcdParserSingle {
    fn translate(log: String) -> (String, String) {
        (String::from("English"), log)
    }
}

impl IntegrityChecker for EzcdParserSingle {}

impl EzcdParserTrack {
    fn new(num: u8, is_range: bool, pregap: Option<Time>, preemphasis: Option<bool>, raw: String) -> Self {
        EzcdParserTrack { num, is_range, pregap, preemphasis, raw }
    }

    fn string_match(&self, regex: &Regex) -> String {
        match regex.captures(&self.raw) {
            Some(val) => val.name("value").unwrap().as_str().trim().to_string(),
            None => String::default(),
        }
    }
}

impl ParserTrack for EzcdParserTrack {}

impl TrackExtractor for EzcdParserTrack {
    fn extract_num(&self) -> u8 {
        self.num
    }

    fn extract_is_range(&self) -> bool {
        self.is_range
    }

    fn extract_is_aborted(&self) -> bool {
        !FINISHED.is_match(&self.raw)
    }

    fn extract_filenames(&self) -> Vec<String> {
        let filename = self.string_match(&FILENAME);
        if filename.is_empty() { Vec::new() } else { vec![filename] }
    }

    fn extract_pregap_length(&self) -> Option<Time> {
        self.pregap
    }

    fn extract_preemphasis(&self) -> Option<bool> {
        self.preemphasis
    }

    fn extract_test_and_copy(&self) -> TestAndCopy {
        TestAndCopy::new_no_skipzero(self.string_match(&TEST_CRC), self.string_match(&COPY_CRC))
    }

    fn extract_errors(&self) -> TrackError {
        let mut sectors: Vec<(&str, u32)> = Vec::new();

        for captures in SECTOR_ERROR.captures_iter(&self.raw) {
            match (captures.name("kind"), captures.name("at")) {
                (Some(kind), Some(at)) => sectors.push((kind.as_str(), frames(at.as_str()))),
                _ => {
                    let pos = frames(&captures["pos"]);
                    for kind in ERROR_KIND.captures_iter(captures.name("kinds").unwrap().as_str()) {
                        sectors.push((kind.get(1).unwrap().as_str(), pos));
                    }
                },
            }
        }

        let errors = |error_type: &str| error_data(sectors.iter().filter(|(kind, _)| *kind == error_type).map(|(_, pos)| *pos).collect());

        TrackError::new_ezcd(errors("Read"), errors("Synchronization"), errors("Data"), errors("CRC"))
    }
}

fn frames(msf: &str) -> u32 {
    let mut parts = msf.split(':').map(|part| part.parse::<u32>().unwrap_or_default());
    let (m, s, f) = (parts.next().unwrap_or_default(), parts.next().unwrap_or_default(), parts.next().unwrap_or_default());
    (m * 60 + s) * 75 + f
}

fn error_data(mut positions: Vec<u32>) -> TrackErrorData {
    let count = positions.len() as u32;
    positions.sort_unstable();
    positions.dedup();

    let mut spans: Vec<(u32, u32)> = Vec::new();
    for pos in positions {
        match spans.last_mut() {
            Some((_, end)) if *end + 1 == pos => *end = pos,
            _ => spans.push((pos, pos)),
        }
    }

    let ranges = spans.into_iter().map(|(start, end)| TrackErrorRange::new_from_end(Time::from_frames(start), Time::from_frames(end))).collect();
    TrackErrorData::new(count, ranges)
}
