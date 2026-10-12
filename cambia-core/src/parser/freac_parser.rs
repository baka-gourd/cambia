use std::str::FromStr;

use regex::Regex;

use crate::{extract::{Extractor, ReleaseInfo, Ripper, TrackExtractor}, integrity::IntegrityChecker, toc::{Toc, TocEntry, TocRaw}, track::{AccurateRipConfidence, AccurateRipOffset, AccurateRipStatus, AccurateRipUnit, TestAndCopy, TrackEntry}, translate::{Translator, TranslatorCombined}, util::Time};
use simple_text_decode::DecodedText;

use super::{split_tracks, Parser, ParsedLog, ParserCombined, ParsedLogCombined, ParserTrack};

lazy_static! {
    static ref RIPPER_VERSION: Regex = Regex::new(r"Version:\s+v?(\S+)").unwrap();
    static ref USED_DRIVE: Regex = Regex::new(r"(?m)Drive model:\s+(?P<value>.+?)\s*$").unwrap();
    static ref READ_OFFSET: Regex = Regex::new(r"Read offset:\s+(?P<value>[+-]?\d+) samples").unwrap();
    static ref ENCODER: Regex = Regex::new(r"(?m)Selected encoder:\s+(?P<value>.+?)\s*$").unwrap();
    static ref ARTIST: Regex = Regex::new(r"(?m)Artist:\s+(?P<value>.+?)\s*$").unwrap();
    static ref ALBUM: Regex = Regex::new(r"(?m)Album:\s+(?P<value>.+?)\s*$").unwrap();
    static ref TOC: Regex = Regex::new(r"(?m)^\S+\s+(?P<track>\d+) \| (?P<start>[\d:.]+) \| (?P<length>[\d:.]+) \|\s+(?P<start_sector>\d+) \|\s+(?P<end_sector>\d+)").unwrap();

    static ref SPLIT_TRACKS: Regex = Regex::new(r"Ripping: device://cdda:\d+/(\d+)").unwrap();
    static ref FILENAME: Regex = Regex::new(r"(?m)^\S+\s+to: (?P<value>.+?)\s*$").unwrap();
    static ref CRC: Regex = Regex::new(r"CRC checksum: (?P<value>[0-9a-fA-F]+)").unwrap();
    static ref AR: Regex = Regex::new(r"Checksum \(AccurateRip v(?P<version>\d+)\): (?P<sign>[0-9a-fA-F]+), Confidence: (?P<cm>\d+)").unwrap();
    static ref AR_FOUND: Regex = Regex::new(r"Track has been accurately ripped").unwrap();
    static ref EXTRACTION_SPEED: Regex = Regex::new(r"Duration: [\d:.]+ \((?P<value>[\d.]+)x speed\)").unwrap();
}

pub struct FreacParser {
    encoded_log: DecodedText,
}

struct FreacParserSingle {
    log: String,
    language: String,
}

struct FreacParserTrack {
    num: u8,
    raw: String,
}

impl FreacParser {
    pub fn new(encoded_log: DecodedText) -> FreacParser {
        FreacParser {
            encoded_log,
        }
    }
}

impl FreacParserSingle {
    pub fn new(log: String) -> FreacParserSingle {
        let (language, _) = FreacParserSingle::translate(log.clone());
        FreacParserSingle {
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

    fn tracks(&self) -> Vec<FreacParserTrack> {
        split_tracks(&self.log, &SPLIT_TRACKS)
            .into_iter()
            .map(|(num, raw)| FreacParserTrack::new(num, raw.to_owned()))
            .collect()
    }
}

impl ParserCombined for FreacParser {
    fn parse_combined(&self) -> ParsedLogCombined {
        let parsed_logs: Vec<ParsedLog> = vec![FreacParserSingle::new(self.encoded_log.text.trim().to_string()).parse()];

        ParsedLogCombined {
            parsed_logs,
            encoding: self.encoded_log.orig_encoding.to_string()
        }
    }
}

impl TranslatorCombined for FreacParser {
    fn translate_combined(&self) -> String {
        self.encoded_log.text.clone()
    }
}

impl Parser for FreacParserSingle {}

impl Extractor for FreacParserSingle {
    fn extract_ripper(&self) -> Ripper {
        Ripper::FreAc
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
        let artist = self.string_match(&ARTIST);
        let album = self.string_match(&ALBUM);
        if artist.is_empty() || album.is_empty() { ReleaseInfo::default() } else { ReleaseInfo::new(artist, album) }
    }

    fn extract_drive(&self) -> String {
        self.string_match(&USED_DRIVE)
    }

    fn extract_read_offset(&self) -> Option<i16> {
        self.optional_match::<i16>(&READ_OFFSET)
    }

    fn extract_audio_encoder(&self) -> Vec<String> {
        let encoder = self.string_match(&ENCODER);
        if encoder.is_empty() { Vec::new() } else { vec![encoder] }
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

impl Translator for FreacParserSingle {
    fn translate(log: String) -> (String, String) {
        (String::from("English"), log)
    }
}

impl IntegrityChecker for FreacParserSingle {}

impl FreacParserTrack {
    fn new(num: u8, raw: String) -> Self {
        FreacParserTrack { num, raw }
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

impl ParserTrack for FreacParserTrack {}

impl TrackExtractor for FreacParserTrack {
    fn extract_num(&self) -> u8 {
        self.num
    }

    fn extract_is_range(&self) -> bool {
        false
    }

    fn extract_is_aborted(&self) -> bool {
        !CRC.is_match(&self.raw)
    }

    fn extract_filenames(&self) -> Vec<String> {
        let filename = self.string_match(&FILENAME);
        if filename.is_empty() { Vec::new() } else { vec![filename] }
    }

    fn extract_extraction_speed(&self) -> Option<f64> {
        self.optional_match::<f64>(&EXTRACTION_SPEED)
    }

    fn extract_test_and_copy(&self) -> TestAndCopy {
        TestAndCopy::new_no_skipzero(String::new(), self.string_match(&CRC).to_uppercase())
    }

    fn extract_ar_info(&self) -> Vec<AccurateRipUnit> {
        let Some(captures) = AR.captures(&self.raw) else {
            return Vec::new();
        };

        let sign = captures["sign"].to_uppercase();
        let status = if AR_FOUND.is_match(&self.raw) { AccurateRipStatus::Match } else { AccurateRipStatus::Mismatch };
        let confidence = captures["cm"].parse::<u32>().ok().map(|cm| AccurateRipConfidence::new(Some(cm), None, AccurateRipOffset::Same));

        vec![AccurateRipUnit::new(captures["version"].parse::<u8>().ok(), sign.clone(), sign, confidence, status)]
    }
}
