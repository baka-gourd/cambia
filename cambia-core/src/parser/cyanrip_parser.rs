mod fun512;

use std::str::FromStr;

use regex::Regex;

use crate::{extract::{Extractor, Gap, Quartet, ReadMode, ReleaseInfo, Ripper, TrackExtractor}, integrity::IntegrityChecker, toc::{Toc, TocEntry, TocRaw}, track::{AccurateRipConfidence, AccurateRipOffset, AccurateRipStatus, AccurateRipUnit, TestAndCopy, TrackEntry}, translate::{Translator, TranslatorCombined}, util::Time};
use simple_text_decode::DecodedText;

use super::{split_tracks, Parser, ParsedLog, ParserCombined, ParsedLogCombined, ParserTrack};

lazy_static! {
    static ref RIPPER_VERSION: Regex = Regex::new(r"^cyanrip ([^\s(]+)").unwrap();
    // Only "Drive used" is shaped like the AccurateRip entries, but no release emits it yet
    static ref USED_DRIVE: Regex = Regex::new(r"Drive used:\s+(?P<value>.+)").unwrap();
    static ref DEVICE_MODEL: Regex = Regex::new(r"Device model:\s+(?P<value>.+)").unwrap();
    static ref READ_OFFSET: Regex = Regex::new(r"Offset:\s+(?P<value>[+-]?\d+) samples?").unwrap();
    static ref OVERREAD_MODE: Regex = Regex::new(r"(?:Over|Under)read mode:\s+(.+)").unwrap();
    static ref PARANOIA: Regex = Regex::new(r"Paranoia level:\s+(\S+)").unwrap();
    static ref OUTPUTS: Regex = Regex::new(r"Outputs:\s+(.+)").unwrap();
    static ref ALBUM: Regex = Regex::new(r"(?m)^Album:\s+(?P<value>.+?)\s*$").unwrap();
    static ref ALBUM_ARTIST: Regex = Regex::new(r"(?m)^Album artist:\s+(?P<value>.+?)\s*$").unwrap();

    static ref SPLIT_TRACKS: Regex = Regex::new(r"(?m)^Track (\d+) ripped").unwrap();
    static ref TRACK_OK: Regex = Regex::new(r"^Track \d+ ripped and encoded successfully").unwrap();
    static ref DURATION: Regex = Regex::new(r"Duration:\s+(?P<value>[0-9:.]+)").unwrap();
    static ref START_LSN: Regex = Regex::new(r"Start LSN:\s+(?P<value>\d+)").unwrap();
    static ref END_LSN: Regex = Regex::new(r"End LSN:\s+(?P<value>\d+)").unwrap();
    static ref PREGAP: Regex = Regex::new(r"Pregap LSN:\s+\d+ \(duration: (?P<time>[0-9:.]+)\)").unwrap();
    static ref PREEMPHASIS: Regex = Regex::new(r"(?m)^\s*Preemphasis:\s+(?P<value>.+?)\s*$").unwrap();
    static ref COPY_CRC: Regex = Regex::new(r"EAC CRC32:\s+(?P<value>[0-9A-F]{8})").unwrap();
    static ref FILES_START: Regex = Regex::new(r"(?m)^\s*File\(s\):\s*$").unwrap();
    static ref LABEL: Regex = Regex::new(r"^[A-Za-z][A-Za-z0-9 ()]*:").unwrap();
    static ref GAIN: Regex = Regex::new(r"REPLAYGAIN_TRACK_GAIN:\s+(?P<value>[+-]?[0-9.]+) dB").unwrap();
    static ref PEAK_LEVEL: Regex = Regex::new(r"REPLAYGAIN_TRACK_PEAK:\s+(?P<value>[0-9.]+)").unwrap();

    // v0.5.x printed 0x-prefixed lowercase and had no status line at all
    static ref AR_STATUS: Regex = Regex::new(r"(?m)^\s*Accurip:\s+(?P<value>.+?)\s*$").unwrap();
    static ref AR_SIGNS: Regex = Regex::new(r"(?m)^\s*Accurip v(?P<version>[12]):\s+(?:0x)?(?P<sign>[0-9a-fA-F]{8})(?P<rest>.*)$").unwrap();
    static ref AR_STATUS_CONFIDENCE: Regex = Regex::new(r"\((?:max )?confidence: (\d+)\)").unwrap();
    static ref AR_FOUND: Regex = Regex::new(r"accurately ripped(?:, confidence (\d+))?").unwrap();
    static ref AR_DB_SIGN: Regex = Regex::new(r"Accurip DB of 0x([0-9a-fA-F]+)").unwrap();
}

pub struct CyanRipParser {
    encoded_log: DecodedText,
}

struct CyanRipParserSingle {
    log: String,
    language: String,
}

struct CyanRipParserTrack {
    num: u8,
    raw: String,
}

impl CyanRipParser {
    pub fn new(encoded_log: DecodedText) -> CyanRipParser {
        CyanRipParser {
            encoded_log,
        }
    }
}

impl CyanRipParserSingle {
    pub fn new(log: String) -> CyanRipParserSingle {
        let (language, _) = CyanRipParserSingle::translate(log.clone());
        CyanRipParserSingle {
            log,
            language,
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

    fn tracks(&self) -> Vec<CyanRipParserTrack> {
        split_tracks(&self.log, &SPLIT_TRACKS)
            .into_iter()
            .map(|(num, raw)| CyanRipParserTrack::new(num, raw.to_owned()))
            .collect()
    }
}

impl ParserCombined for CyanRipParser {
    fn parse_combined(&self) -> ParsedLogCombined {
        let parsed_logs: Vec<ParsedLog> = vec![CyanRipParserSingle::new(self.encoded_log.text.trim().to_string()).parse()];

        ParsedLogCombined {
            parsed_logs,
            encoding: self.encoded_log.orig_encoding.to_string()
        }
    }
}

impl TranslatorCombined for CyanRipParser {
    fn translate_combined(&self) -> String {
        self.encoded_log.text.clone()
    }
}

impl Parser for CyanRipParserSingle {}

impl Extractor for CyanRipParserSingle {
    fn extract_ripper(&self) -> Ripper {
        Ripper::CyanRip
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
        let artist = self.string_match(&ALBUM_ARTIST);
        let album = self.string_match(&ALBUM);
        if artist.is_empty() || album.is_empty() { ReleaseInfo::default() } else { ReleaseInfo::new(artist, album) }
    }

    fn extract_drive(&self) -> String {
        let drive = self.string_match(&USED_DRIVE);
        if drive.is_empty() { self.string_match(&DEVICE_MODEL) } else { drive }
    }

    fn extract_read_offset(&self) -> Option<i16> {
        self.optional_match::<i16>(&READ_OFFSET)
    }

    fn extract_overread(&self) -> Quartet {
        let captures = OVERREAD_MODE.captures(&self.log);
        match captures {
            Some(captures) => {
                let value = captures.get(1).unwrap().as_str().trim();
                if value.starts_with("read in") { Quartet::True } else { Quartet::False }
            },
            None => Quartet::Unknown,
        }
    }

    fn extract_fill_silence(&self) -> Quartet {
        let captures = OVERREAD_MODE.captures(&self.log);
        match captures {
            Some(captures) => {
                let value = captures.get(1).unwrap().as_str().trim();
                if value.starts_with("fill with silence") { Quartet::True } else { Quartet::False }
            },
            None => Quartet::Unknown,
        }
    }

    fn extract_read_mode(&self) -> ReadMode {
        let captures = PARANOIA.captures(&self.log);
        match captures {
            Some(captures) => {
                let value = captures.get(1).unwrap().as_str().trim();
                match value {
                    "max" => ReadMode::Paranoid,
                    "none" => ReadMode::Burst,
                    _ => ReadMode::Secure,
                }
            },
            None => ReadMode::Unknown,
        }
    }

    fn extract_gap_handling(&self) -> Gap {
        Gap::AppendNoHtoa
    }

    fn extract_audio_encoder(&self) -> Vec<String> {
        let captures = OUTPUTS.captures(&self.log);
        match captures {
            Some(captures) => captures.get(1).unwrap().as_str().trim().split(',').map(|output| output.trim().to_owned()).collect(),
            None => Vec::new(),
        }
    }

    fn extract_toc(&self) -> Toc {
        let entries: Vec<TocEntry> = self.tracks().iter().filter_map(|track| {
            let start = track.optional_match::<u32>(&START_LSN)?;
            let end = track.optional_match::<u32>(&END_LSN)?;
            let duration = track.string_match(&DURATION);
            let length = if duration.is_empty() { Time::from_ss("0") } else { Time::from_h_mm_ss(&duration) };
            Some(TocEntry::new(
                u32::from(track.num),
                Time::from_frames(start),
                length,
                start,
                end,
            ))
        }).collect();

        Toc::new(TocRaw::new(entries))
    }

    fn extract_tracks(&self) -> Vec<TrackEntry> {
        self.tracks().iter().map(|track| track.parse_track()).collect()
    }
}

impl Translator for CyanRipParserSingle {
    fn translate(log: String) -> (String, String) {
        (String::from("English"), log)
    }
}

impl IntegrityChecker for CyanRipParserSingle {
    fn extract_checksum(&self) -> String {
        fun512::extract(&self.log).unwrap_or_default().to_owned()
    }

    fn calculate_checksum(&self) -> String {
        fun512::calculate(&self.log)
    }
}

impl CyanRipParserTrack {
    fn new(num: u8, raw: String) -> Self {
        CyanRipParserTrack { num, raw }
    }

    fn string_match(&self, regex: &Regex) -> String {
        match regex.captures(&self.raw) {
            Some(val) => val.name("value").unwrap().as_str().trim().to_string(),
            None => String::default(),
        }
    }

    fn optional_match<T>(&self, regex: &Regex) -> Option<T>
    where
        T: FromStr, {
        regex.captures(&self.raw).and_then(|val| {
            val.name("value").and_then(|v| v.as_str().trim().parse::<T>().ok())
        })
    }
}

impl ParserTrack for CyanRipParserTrack {}

impl TrackExtractor for CyanRipParserTrack {
    fn extract_num(&self) -> u8 {
        self.num
    }

    fn extract_is_range(&self) -> bool {
        false
    }

    fn extract_is_aborted(&self) -> bool {
        !TRACK_OK.is_match(&self.raw)
    }

    fn extract_filenames(&self) -> Vec<String> {
        let Some(m) = FILES_START.find(&self.raw) else {
            return Vec::new();
        };

        self.raw[m.end()..]
            .lines()
            .skip(1)
            .map(str::trim)
            .take_while(|line| !line.is_empty() && !LABEL.is_match(line))
            .map(str::to_owned)
            .collect()
    }

    fn extract_peak_level(&self) -> Option<f64> {
        self.optional_match::<f64>(&PEAK_LEVEL)
    }

    fn extract_pregap_length(&self) -> Option<Time> {
        let captures = PREGAP.captures(&self.raw);
        captures.map(|captures| Time::from_h_mm_ss(captures.name("time").unwrap().as_str()))
    }

    fn extract_gain(&self) -> Option<f64> {
        self.optional_match::<f64>(&GAIN)
    }

    fn extract_preemphasis(&self) -> Option<bool> {
        let captures = PREEMPHASIS.captures(&self.raw);
        captures.map(|captures| !captures["value"].trim().starts_with("none"))
    }

    fn extract_test_and_copy(&self) -> TestAndCopy {
        TestAndCopy::new_no_skipzero(String::new(), self.string_match(&COPY_CRC))
    }

    fn extract_ar_info(&self) -> Vec<AccurateRipUnit> {
        let state = self.string_match(&AR_STATUS);
        let disabled = state.starts_with("disabled");
        let found = state.starts_with("found") || state.starts_with("disc found");
        let status_confidence = AR_STATUS_CONFIDENCE.captures(&state).and_then(|captures| captures[1].parse::<u32>().ok());

        AR_SIGNS.captures_iter(&self.raw).map(|sign_line| {
            let sign = sign_line["sign"].to_uppercase();
            let rest = &sign_line["rest"];

            let (status, matching) = match AR_FOUND.captures(rest) {
                // v0.6.0 reported confidence once on the status line instead
                Some(ar_f_raw) => (AccurateRipStatus::Match, ar_f_raw.get(1).and_then(|cm| cm.as_str().parse::<u32>().ok()).or(status_confidence)),
                None if disabled => (AccurateRipStatus::Disabled, None),
                // An unannotated checksum means the other version matched
                None if found => (AccurateRipStatus::Mismatch, None),
                None => (AccurateRipStatus::NotFound, None),
            };

            let offset_sign = match status {
                AccurateRipStatus::Match => sign.clone(),
                _ => AR_DB_SIGN.captures(rest).map(|db| format!("{:0>8}", db[1].to_uppercase())).unwrap_or_default(),
            };

            AccurateRipUnit::new(
                sign_line["version"].parse::<u8>().ok(),
                sign,
                offset_sign,
                matching.map(|cm| AccurateRipConfidence::new(Some(cm), None, AccurateRipOffset::Same)),
                status,
            )
        }).collect()
    }
}
