use std::str::FromStr;

use regex::Regex;

use crate::{extract::{Extractor, Gap, Quartet, ReadMode, ReleaseInfo, Ripper, TrackExtractor}, integrity::IntegrityChecker, toc::{Toc, TocEntry, TocRaw}, track::{AccurateRipUnit, TestAndCopy, TrackEntry}, translate::{Translator, TranslatorCombined}, util::Time};
use simple_text_decode::DecodedText;

use super::{split_tracks, Parser, ParsedLog, ParserCombined, ParsedLogCombined, ParserTrack};

lazy_static! {
    static ref RIPPER_VERSION: Regex = Regex::new(r"Logfile created by: morituri (\S+)").unwrap();
    static ref RELEASE_INFO: Regex = Regex::new(r"(?m)^Album: (?P<artist>.+?) - (?P<title>.+?)\s*$").unwrap();
    static ref USED_DRIVE: Regex = Regex::new(r"(?m)^Drive: vendor (?P<vendor>.+?), model (?P<model>.+?)\s*$").unwrap();
    static ref READ_OFFSET_CORRECTION: Regex = Regex::new(r"Read offset correction: (?P<value>[+-]?\d+)").unwrap();
    static ref TOC: Regex = Regex::new(r"(?m)^\s+(?P<track>\d+)\s+\|\s+(?P<start>\d+) - \S+ \|\s+(?P<length>\d+) - \S+").unwrap();
    static ref NO_CRC_CHECK: Regex = Regex::new(r"WARNING: no CRC check done").unwrap();

    static ref SPLIT_TRACKS: Regex = Regex::new(r"(?m)^Track\s+(\d+)\s*$").unwrap();
    static ref FILENAME: Regex = Regex::new(r"(?m)^\s+Filename (?P<value>.+?)\s*$").unwrap();
    static ref PREGAP: Regex = Regex::new(r"Pre-gap: (?P<time>\S+)").unwrap();
    static ref PEAK_LEVEL: Regex = Regex::new(r"Peak level (?P<value>[0-9.]+) %").unwrap();
    static ref EXTRACTION_SPEED: Regex = Regex::new(r"Extraction Speed \(Copy\) (?P<value>[0-9.]+) X").unwrap();
    static ref TEST_CRC: Regex = Regex::new(r"Test CRC (?P<value>[0-9A-F]{8})").unwrap();
    static ref COPY_CRC: Regex = Regex::new(r"Copy CRC (?P<value>[0-9A-F]{8})").unwrap();
    static ref AR_FOUND: Regex = Regex::new(r"Accurately ripped \(confidence (?P<cm>\d+)\) \[(?P<sign>[0-9A-F]{8})\]").unwrap();
    static ref AR_MISMATCH: Regex = Regex::new(r"Cannot be verified as accurate \[(?P<sign>[0-9A-F]{8})\], AccurateRip returned \[(?P<off_sign>[0-9A-F]{8})\]").unwrap();
    static ref AR_NO_DB: Regex = Regex::new(r"Track not present in AccurateRip database").unwrap();
}

pub struct MorituriParser {
    encoded_log: DecodedText,
}

struct MorituriParserSingle {
    log: String,
    language: String,
}

struct MorituriParserTrack {
    num: u8,
    raw: String,
}

impl MorituriParser {
    pub fn new(encoded_log: DecodedText) -> MorituriParser {
        MorituriParser {
            encoded_log,
        }
    }
}

impl MorituriParserSingle {
    pub fn new(log: String) -> MorituriParserSingle {
        let (language, _) = MorituriParserSingle::translate(log.clone());
        MorituriParserSingle {
            log,
            language,
        }
    }

    fn tracks(&self) -> Vec<MorituriParserTrack> {
        split_tracks(&self.log, &SPLIT_TRACKS)
            .into_iter()
            .map(|(num, raw)| MorituriParserTrack::new(num, raw.to_owned()))
            .collect()
    }
}

impl ParserCombined for MorituriParser {
    fn parse_combined(&self) -> ParsedLogCombined {
        let parsed_logs: Vec<ParsedLog> = vec![MorituriParserSingle::new(self.encoded_log.text.trim().to_string()).parse()];

        ParsedLogCombined {
            parsed_logs,
            encoding: self.encoded_log.orig_encoding.to_string()
        }
    }
}

impl TranslatorCombined for MorituriParser {
    fn translate_combined(&self) -> String {
        self.encoded_log.text.clone()
    }
}

impl Parser for MorituriParserSingle {}

impl Extractor for MorituriParserSingle {
    fn extract_ripper(&self) -> Ripper {
        Ripper::Morituri
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
            Some(captures) => ReleaseInfo::new(captures["artist"].trim().to_owned(), captures["title"].trim().to_owned()),
            None => ReleaseInfo::default(),
        }
    }

    fn extract_drive(&self) -> String {
        let captures = USED_DRIVE.captures(&self.log);
        match captures {
            Some(captures) => format!("{} {}", captures["vendor"].trim(), captures["model"].trim()),
            None => String::default(),
        }
    }

    fn extract_read_offset(&self) -> Option<i16> {
        let captures = READ_OFFSET_CORRECTION.captures(&self.log);
        captures.and_then(|captures| captures["value"].parse::<i16>().ok())
    }

    fn extract_read_mode(&self) -> ReadMode {
        ReadMode::Secure
    }

    fn extract_accurate_stream(&self) -> Quartet {
        Quartet::True
    }

    fn extract_defeat_audio_cache(&self) -> Quartet {
        Quartet::Unknown
    }

    fn extract_use_null_samples(&self) -> Quartet {
        Quartet::True
    }

    fn extract_gap_handling(&self) -> Gap {
        Gap::Append
    }

    fn extract_test_and_copy(&self) -> Quartet {
        if NO_CRC_CHECK.is_match(&self.log) {
            Quartet::False
        } else if TEST_CRC.is_match(&self.log) {
            Quartet::True
        } else {
            Quartet::Unknown
        }
    }

    fn extract_toc(&self) -> Toc {
        let mut entries: Vec<TocEntry> = Vec::new();
        let captures_all = TOC.captures_iter(&self.log);

        for captures in captures_all {
            let Ok(track) = captures["track"].parse::<u32>() else { continue };
            let Ok(start) = captures["start"].parse::<u32>() else { continue };
            let Ok(length) = captures["length"].parse::<u32>() else { continue };

            entries.push(TocEntry::new(
                track,
                Time::from_frames(start),
                Time::from_frames(length),
                start,
                start + length - 1,
            ));
        }

        Toc::new(TocRaw::new(entries))
    }

    fn extract_tracks(&self) -> Vec<TrackEntry> {
        self.tracks().iter().map(|track| track.parse_track()).collect()
    }
}

impl Translator for MorituriParserSingle {
    fn translate(log: String) -> (String, String) {
        (String::from("English"), log)
    }
}

impl IntegrityChecker for MorituriParserSingle {}

impl MorituriParserTrack {
    fn new(num: u8, raw: String) -> Self {
        MorituriParserTrack { num, raw }
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

impl ParserTrack for MorituriParserTrack {}

impl TrackExtractor for MorituriParserTrack {
    fn extract_num(&self) -> u8 {
        self.num
    }

    fn extract_is_range(&self) -> bool {
        false
    }

    fn extract_filenames(&self) -> Vec<String> {
        let filename = self.string_match(&FILENAME);
        if filename.is_empty() { Vec::new() } else { vec![filename] }
    }

    fn extract_peak_level(&self) -> Option<f64> {
        self.optional_match::<f64>(&PEAK_LEVEL).map(|val| val / 100.0)
    }

    fn extract_pregap_length(&self) -> Option<Time> {
        let captures = PREGAP.captures(&self.raw);
        captures.map(|captures| Time::from_mm_ss_cs(captures.name("time").unwrap().as_str()))
    }

    fn extract_extraction_speed(&self) -> Option<f64> {
        self.optional_match(&EXTRACTION_SPEED)
    }

    fn extract_test_and_copy(&self) -> TestAndCopy {
        let test_crc = self.string_match(&TEST_CRC);
        let copy_crc = self.string_match(&COPY_CRC);

        TestAndCopy::new_no_skipzero(test_crc, copy_crc)
    }

    fn extract_ar_info(&self) -> Vec<AccurateRipUnit> {
        let mut ars: Vec<AccurateRipUnit> = Vec::new();

        if let Some(ar_f_raw) = AR_FOUND.captures(&self.raw) {
            let matching = ar_f_raw["cm"].parse::<u32>().unwrap_or_default();
            ars.push(AccurateRipUnit::new_eac(1, ar_f_raw["sign"].to_owned(), matching));
        } else if let Some(ar_m_raw) = AR_MISMATCH.captures(&self.raw) {
            ars.push(AccurateRipUnit::new_eac_mismatch(1, ar_m_raw["sign"].to_owned(), ar_m_raw["off_sign"].to_owned(), 0));
        } else if AR_NO_DB.is_match(&self.raw) {
            ars.push(AccurateRipUnit::new_eac_notfound());
        }

        ars
    }
}
