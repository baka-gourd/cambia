use std::str::FromStr;

use regex::Regex;

use crate::{extract::{Extractor, Quartet, ReadMode, Ripper, TrackExtractor}, integrity::IntegrityChecker, toc::{Toc, TocEntry, TocRaw}, track::{AccurateRipStatus, AccurateRipUnit, TestAndCopy, TrackEntry}, translate::{Translator, TranslatorCombined}, util::Time};
use simple_text_decode::DecodedText;

use super::{split_tracks, Parser, ParsedLog, ParserCombined, ParsedLogCombined, ParserTrack};

lazy_static! {
    // macOS builds omit the "Release x.y" entirely
    static ref RIPPER_VERSION: Regex = Regex::new(r"dBpoweramp Release (\S+)").unwrap();
    static ref USED_DRIVE: Regex = Regex::new(r"Ripping with drive '[^\[]*\[\s*(?P<vendor>.+?)\s+-\s+(?P<model>.+?)\s*\]'").unwrap();
    static ref READ_OFFSET: Regex = Regex::new(r"Drive offset:\s*(?P<value>[+-]?\d+)").unwrap();
    static ref OVERREAD: Regex = Regex::new(r"Overread Lead-in/out:\s*(?P<boolean>Yes|No)").unwrap();
    static ref USE_C2: Regex = Regex::new(r"Using C2:\s*(?P<boolean>Yes|No)").unwrap();
    static ref ENCODER: Regex = Regex::new(r"(?m)^Encoder:\s*(?P<value>.+?)\s*$").unwrap();
    static ref SECOND_PASS: Regex = Regex::new(r"Pass 2 Drive Speed:").unwrap();
    static ref TWO_PASS_TRACK: Regex = Regex::new(r"\[Pass \d+ & \d+").unwrap();

    static ref SPLIT_TRACKS: Regex = Regex::new(r"(?m)^Track (\d+):").unwrap();
    static ref RIPPED: Regex = Regex::new(r"Ripped LBA (?P<start>\d+) to (?P<end>\d+) \((?P<len>[0-9:]+)\) in (?P<took>[0-9:]+)\.").unwrap();
    static ref FILENAME: Regex = Regex::new(r"(?m)Filename: (?P<value>.+?)\s*$").unwrap();
    static ref CRC: Regex = Regex::new(r"CRC32: (?P<value>[0-9A-Fa-f]+)").unwrap();
    // The AccurateRip CRC here is zero-padded, the one on the Verified line is not
    static ref AR_CRC: Regex = Regex::new(r"AccurateRip CRC: (?P<value>[0-9A-Fa-f]+)(?: \(CRCv(?P<ver>\d+)\))?").unwrap();
    static ref AR_STATUS: Regex = Regex::new(r"AccurateRip: (?P<verdict>Accurate|Inaccurate) \(confidence (?P<conf>\d+)\)").unwrap();
    static ref AR_VERIFIED: Regex = Regex::new(r"AccurateRip Verified Confidence (?P<conf>\d+) \[CRCv(?P<ver>\d+) (?P<crc>[0-9A-Fa-f]+)\](?:, Using Pressing Offset (?P<offset>[+-]\d+))?").unwrap();
}

fn mm_ss_to_secs(mm_ss: &str) -> Option<f64> {
    let (m, s) = mm_ss.split_once(':')?;
    Some(m.parse::<f64>().ok()? * 60.0 + s.parse::<f64>().ok()?)
}

pub struct DbpaParser {
    encoded_log: DecodedText,
}

struct DbpaParserSingle {
    log: String,
    language: String,
}

struct DbpaParserTrack {
    num: u8,
    raw: String,
}

impl DbpaParser {
    pub fn new(encoded_log: DecodedText) -> DbpaParser {
        DbpaParser {
            encoded_log,
        }
    }
}

impl DbpaParserSingle {
    pub fn new(log: String) -> DbpaParserSingle {
        let (language, _) = DbpaParserSingle::translate(log.clone());
        DbpaParserSingle {
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

    fn optional_match<T>(&self, regex: &Regex) -> Option<T>
    where
        T: FromStr, {
        regex.captures(&self.log).and_then(|val| {
            val.name("value").and_then(|v| v.as_str().trim().parse::<T>().ok())
        })
    }

    fn tracks(&self) -> Vec<DbpaParserTrack> {
        split_tracks(&self.log, &SPLIT_TRACKS)
            .into_iter()
            .map(|(num, raw)| DbpaParserTrack::new(num, raw.to_owned()))
            .collect()
    }
}

impl ParserCombined for DbpaParser {
    fn parse_combined(&self) -> ParsedLogCombined {
        let parsed_logs: Vec<ParsedLog> = vec![DbpaParserSingle::new(self.encoded_log.text.trim().to_string()).parse()];

        ParsedLogCombined {
            parsed_logs,
            encoding: self.encoded_log.orig_encoding.to_string()
        }
    }
}

impl TranslatorCombined for DbpaParser {
    fn translate_combined(&self) -> String {
        self.encoded_log.text.clone()
    }
}

impl Parser for DbpaParserSingle {}

impl Extractor for DbpaParserSingle {
    fn extract_ripper(&self) -> Ripper {
        Ripper::DBPA
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

    fn extract_drive(&self) -> String {
        let captures = USED_DRIVE.captures(&self.log);
        match captures {
            Some(captures) => format!("{} {}", captures["vendor"].trim(), captures["model"].trim()),
            None => String::default(),
        }
    }

    fn extract_read_offset(&self) -> Option<i16> {
        self.optional_match::<i16>(&READ_OFFSET)
    }

    fn extract_overread(&self) -> Quartet {
        self.boolean_matcher(&OVERREAD)
    }

    fn extract_use_c2(&self) -> Quartet {
        self.boolean_matcher(&USE_C2)
    }

    fn extract_read_mode(&self) -> ReadMode {
        let captures = SECOND_PASS.captures(&self.log);
        match captures {
            Some(_) => ReadMode::Secure,
            None => ReadMode::Unknown,
        }
    }

    fn extract_test_and_copy(&self) -> Quartet {
        let captures = TWO_PASS_TRACK.captures(&self.log);
        match captures {
            Some(_) => Quartet::True,
            None => Quartet::False,
        }
    }

    fn extract_audio_encoder(&self) -> Vec<String> {
        let captures = ENCODER.captures(&self.log);
        match captures {
            Some(captures) => vec![captures["value"].split_whitespace().next().unwrap_or_default().to_owned()],
            None => Vec::new(),
        }
    }

    fn extract_toc(&self) -> Toc {
        let entries: Vec<TocEntry> = self.tracks().iter().filter_map(|track| {
            let captures = RIPPED.captures(&track.raw)?;
            let start = captures["start"].parse::<u32>().ok()?;
            let end = captures["end"].parse::<u32>().ok()?;
            Some(TocEntry::new(
                u32::from(track.num),
                Time::from_frames(start),
                Time::from_frames(end.saturating_sub(start)),
                start,
                end.saturating_sub(1),
            ))
        }).collect();

        Toc::new(TocRaw::new(entries))
    }

    fn extract_tracks(&self) -> Vec<TrackEntry> {
        self.tracks().iter().map(|track| track.parse_track()).collect()
    }
}

impl Translator for DbpaParserSingle {
    fn translate(log: String) -> (String, String) {
        (String::from("English"), log)
    }
}

impl IntegrityChecker for DbpaParserSingle {}

impl DbpaParserTrack {
    fn new(num: u8, raw: String) -> Self {
        DbpaParserTrack { num, raw }
    }

    fn string_match(&self, regex: &Regex) -> String {
        match regex.captures(&self.raw) {
            Some(val) => val.name("value").unwrap().as_str().trim().to_string(),
            None => String::default(),
        }
    }
}

impl ParserTrack for DbpaParserTrack {}

impl TrackExtractor for DbpaParserTrack {
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
        let captures = RIPPED.captures(&self.raw)?;
        let len = mm_ss_to_secs(&captures["len"])?;
        let took = mm_ss_to_secs(&captures["took"])?;
        if took > 0.0 { Some(len / took) } else { None }
    }

    fn extract_test_and_copy(&self) -> TestAndCopy {
        TestAndCopy::new_no_skipzero(String::new(), self.string_match(&CRC).to_uppercase())
    }

    fn extract_ar_info(&self) -> Vec<AccurateRipUnit> {
        let version = AR_VERIFIED.captures(&self.raw)
            .and_then(|captures| captures.name("ver"))
            .or_else(|| AR_CRC.captures(&self.raw).and_then(|captures| captures.name("ver")))
            .and_then(|m| m.as_str().parse::<u8>().ok())
            .unwrap_or(1);
        let crc = self.string_match(&AR_CRC).to_uppercase();

        match AR_STATUS.captures(&self.raw) {
            Some(captures) => {
                let matching = captures["conf"].parse::<u32>().unwrap_or_default();
                match &captures["verdict"] {
                    "Accurate" => vec![AccurateRipUnit::new_eac(version, crc, matching)],
                    _ => vec![AccurateRipUnit::new_eac_mismatch(version, crc.clone(), crc, matching)],
                }
            },
            None if !crc.is_empty() => vec![AccurateRipUnit::new(Some(version), crc.clone(), crc, None, AccurateRipStatus::NotFound)],
            None => Vec::new(),
        }
    }
}
