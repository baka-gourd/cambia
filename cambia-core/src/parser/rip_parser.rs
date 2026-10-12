use std::str::FromStr;

use regex::Regex;

use crate::{extract::{Extractor, Quartet, ReleaseInfo, Ripper, TrackExtractor}, integrity::IntegrityChecker, toc::{Toc, TocEntry, TocRaw}, track::{AccurateRipConfidence, AccurateRipOffset, AccurateRipStatus, AccurateRipUnit, TrackEntry}, translate::{Translator, TranslatorCombined}, util::Time};
use simple_text_decode::DecodedText;

use super::{split_tracks, Parser, ParsedLog, ParserCombined, ParsedLogCombined, ParserTrack};

lazy_static! {
    static ref RIPPER_VERSION: Regex = Regex::new(r"^Rip (.+?) Audio Extraction Log").unwrap();
    static ref USED_DRIVE: Regex = Regex::new(r"(?m)^Drive used:\s+(?P<value>.+?)\s*$").unwrap();
    static ref DRIVE_TRIM: Regex = Regex::new(r"\s*\([^)]*\)\s*$").unwrap();
    static ref READ_OFFSET: Regex = Regex::new(r"(?m)^Read offset:\s+(?P<value>[+-]?\d+)").unwrap();
    static ref ACCURATE_STREAM: Regex = Regex::new(r"(?m)^Stream accurate:\s+(?P<boolean>Yes|No)").unwrap();
    static ref RELEASE_INFO: Regex = Regex::new(r"(?m)^Name:\s+(?P<artist>.+?) - (?P<title>.+?)\s*$").unwrap();
    static ref COPY_VERIFIED: Regex = Regex::new(r"Copy verified").unwrap();
    static ref TOC: Regex = Regex::new(r"(?m)^\s*(?P<track>\d+)\s*\|\s*(?P<start>[0-9:.]+)\s*\|\s*[0-9:.]+\s*\|\s*(?P<length>[0-9:.]+)\s*\|\s*(?P<start_sector>\d+)\s*\|\s*(?P<end_sector>\d+)\s*\|").unwrap();

    static ref SPLIT_TRACKS: Regex = Regex::new(r"(?m)^Track (\d+) saved to .+").unwrap();
    static ref FILENAME: Regex = Regex::new(r"(?m)^Track \d+ saved to (?P<value>.+?)\s*$").unwrap();
    static ref AR_SIGN: Regex = Regex::new(r"AccurateRip checksum:\s+(?P<value>[0-9a-fA-F]+)").unwrap();
    static ref AR_FOUND: Regex = Regex::new(r"Accurately ripped:\s+(?P<value>Yes|No)").unwrap();
    static ref AR_CONFIDENCE: Regex = Regex::new(r"Confidence level:\s+(?P<value>\d+)").unwrap();
    static ref PEAK_LEVEL: Regex = Regex::new(r"(?m)^\s*Peak level:\s+(?P<value>[0-9.]+)").unwrap();
    static ref GAIN: Regex = Regex::new(r"(?m)^\s*Replay gain:\s+(?P<value>[+-]?[0-9.]+) dB").unwrap();
}

// Rip prints AccurateRip checksums sign-extended to 64 bits
fn low32(hex: &str) -> String {
    format!("{:08X}", u64::from_str_radix(hex, 16).unwrap_or_default() as u32)
}

pub struct RipParser {
    encoded_log: DecodedText,
}

struct RipParserSingle {
    log: String,
    language: String,
}

struct RipParserTrack {
    num: u8,
    raw: String,
}

impl RipParser {
    pub fn new(encoded_log: DecodedText) -> RipParser {
        RipParser {
            encoded_log,
        }
    }
}

impl RipParserSingle {
    pub fn new(log: String) -> RipParserSingle {
        let (language, _) = RipParserSingle::translate(log.clone());
        RipParserSingle {
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
                    "Yes" => Quartet::True,
                    "No" => Quartet::False,
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

    fn tracks(&self) -> Vec<RipParserTrack> {
        split_tracks(&self.log, &SPLIT_TRACKS)
            .into_iter()
            .map(|(num, raw)| RipParserTrack::new(num, raw.to_owned()))
            .collect()
    }
}

impl ParserCombined for RipParser {
    fn parse_combined(&self) -> ParsedLogCombined {
        let parsed_logs: Vec<ParsedLog> = vec![RipParserSingle::new(self.encoded_log.text.trim().to_string()).parse()];

        ParsedLogCombined {
            parsed_logs,
            encoding: self.encoded_log.orig_encoding.to_string()
        }
    }
}

impl TranslatorCombined for RipParser {
    fn translate_combined(&self) -> String {
        self.encoded_log.text.clone()
    }
}

impl Parser for RipParserSingle {}

impl Extractor for RipParserSingle {
    fn extract_ripper(&self) -> Ripper {
        Ripper::Rip
    }

    fn extract_ripper_version(&self) -> String {
        let captures = RIPPER_VERSION.captures(&self.log);
        match captures {
            Some(captures) => captures.get(1).unwrap().as_str().trim().to_string(),
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
        let drive = self.string_match(&USED_DRIVE);
        DRIVE_TRIM.replace(&drive, "").trim().to_string()
    }

    fn extract_read_offset(&self) -> Option<i16> {
        self.optional_match::<i16>(&READ_OFFSET)
    }

    fn extract_accurate_stream(&self) -> Quartet {
        self.boolean_matcher(&ACCURATE_STREAM)
    }

    fn extract_test_and_copy(&self) -> Quartet {
        let captures = COPY_VERIFIED.captures(&self.log);
        match captures {
            Some(_) => Quartet::True,
            None => Quartet::Unknown,
        }
    }

    fn extract_toc(&self) -> Toc {
        let entries: Vec<TocEntry> = TOC.captures_iter(&self.log).filter_map(|captures| {
            Some(TocEntry::new(
                captures["track"].parse::<u32>().ok()?,
                Time::from_mm_ss_ff(&captures["start"]),
                Time::from_mm_ss_ff(&captures["length"]),
                captures["start_sector"].parse::<u32>().ok()?,
                captures["end_sector"].parse::<u32>().ok()?,
            ))
        }).collect();

        Toc::new(TocRaw::new(entries))
    }

    fn extract_tracks(&self) -> Vec<TrackEntry> {
        self.tracks().iter().map(|track| track.parse_track()).collect()
    }
}

impl Translator for RipParserSingle {
    fn translate(log: String) -> (String, String) {
        (String::from("English"), log)
    }
}

impl IntegrityChecker for RipParserSingle {}

impl RipParserTrack {
    fn new(num: u8, raw: String) -> Self {
        RipParserTrack { num, raw }
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

impl ParserTrack for RipParserTrack {}

impl TrackExtractor for RipParserTrack {
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
        self.optional_match::<f64>(&PEAK_LEVEL)
    }

    fn extract_gain(&self) -> Option<f64> {
        self.optional_match::<f64>(&GAIN)
    }

    fn extract_ar_info(&self) -> Vec<AccurateRipUnit> {
        let sign = self.string_match(&AR_SIGN);
        if sign.is_empty() {
            return Vec::new();
        }

        let sign = low32(&sign);
        let confidence = self.optional_match::<u32>(&AR_CONFIDENCE).map(|cm| AccurateRipConfidence::new(Some(cm), None, AccurateRipOffset::Same));
        let status = match self.string_match(&AR_FOUND).as_str() {
            "Yes" => AccurateRipStatus::Match,
            "" => AccurateRipStatus::NotFound,
            _ => AccurateRipStatus::Mismatch,
        };

        vec![AccurateRipUnit::new(None, sign.clone(), sign, confidence, status)]
    }
}
