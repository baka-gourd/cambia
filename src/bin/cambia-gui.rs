#![cfg_attr(target_os = "windows", windows_subsystem = "windows")]

use std::collections::BTreeMap;
use std::fs;
#[cfg(feature = "gui_profiler")]
use std::io::Write;
#[cfg(target_os = "windows")]
use std::os::windows::ffi::OsStrExt;
use std::path::{Path, PathBuf};
use std::sync::{mpsc, Arc, Mutex, OnceLock};
use std::thread;
#[cfg(feature = "gui_profiler")]
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use cambia_core::evaluate::{
    Evaluation, EvaluationCombined, EvaluationUnitClass, EvaluationUnitField, EvaluationUnitScope,
    EvaluatorType,
};
use cambia_core::extract::{Gap, MediaType, Quartet, ReadMode, Ripper};
use cambia_core::handler::parse_log_bytes;
use cambia_core::integrity::Integrity;
use cambia_core::parser::ParsedLog;
use cambia_core::response::CambiaResponse;
use cambia_core::track::{
    AccurateRipConfidenceTotal, AccurateRipOffset, AccurateRipStatus, TrackEntry, TrackErrorData,
};
use cambia_core::util::Time;
use walkdir::WalkDir;
use winio::prelude::*;

#[cfg(target_os = "windows")]
use windows_sys::Win32::Foundation::{HWND, LPARAM, LRESULT, WPARAM};
#[cfg(target_os = "windows")]
use windows_sys::Win32::Graphics::Gdi::{
    CreateFontIndirectW, DeleteObject, GetBkColor, GetTextColor, SetTextColor, FW_BOLD, HDC, HFONT,
    HGDIOBJ, LOGFONTW,
};
#[cfg(target_os = "windows")]
use windows_sys::Win32::UI::Controls::SetScrollInfo;
#[cfg(target_os = "windows")]
use windows_sys::Win32::UI::HiDpi::{
    GetDpiForWindow, GetSystemMetricsForDpi, SystemParametersInfoForDpi,
};
#[cfg(target_os = "windows")]
use windows_sys::Win32::UI::Shell::{
    DragAcceptFiles, DragFinish, DragQueryFileW, ILClone, ILCreateFromPathW, ILFindLastID, ILFree,
    ILRemoveLastID, SHOpenFolderAndSelectItems, HDROP,
};
#[cfg(target_os = "windows")]
use windows_sys::Win32::UI::WindowsAndMessaging::{
    CallWindowProcW, DefWindowProcW, SendMessageW, SetWindowLongPtrW, SetWindowPos, GWLP_WNDPROC,
    LB_ERR, LB_GETCURSEL, NONCLIENTMETRICSW, SB_HORZ, SB_VERT, SCROLLINFO, SIF_POS, SM_CXVSCROLL,
    SM_CYHSCROLL, SPI_GETNONCLIENTMETRICS, SWP_NOACTIVATE, SWP_NOSIZE, SWP_NOZORDER,
    WM_CTLCOLORSTATIC, WM_DROPFILES, WM_SETFONT, WNDPROC,
};

const EMPTY_TEXT: &str = "未加载日志。选择 .log 文件或文件夹，或将 .log / 文件夹拖入窗口。";
const ROW_H: f64 = 22.0;
const SECTION_H: f64 = 28.0;
const DETAIL_PAD: f64 = 8.0;
const OPEN_BUTTON_H: f64 = 28.0;
const OPEN_BUTTON_GAP: f64 = 10.0;
const SCROLLBAR_PAD_EXTRA: f64 = 16.0;
const OVERVIEW_ROW_COUNT: usize = 7;
const LEFT_TEXT_CHARS: usize = 44;

#[derive(Clone, Debug)]
pub struct GuiOptions {
    pub app_name: String,
    #[cfg(target_os = "windows")]
    pub app_user_model_id: Option<String>,
    #[cfg(target_os = "windows")]
    pub icon_resource_id: Option<u16>,
}

impl Default for GuiOptions {
    fn default() -> Self {
        Self {
            app_name: "Cambia".to_string(),
            #[cfg(target_os = "windows")]
            app_user_model_id: Some("Cambia.Cambia".to_string()),
            #[cfg(target_os = "windows")]
            icon_resource_id: Some(101),
        }
    }
}

fn main() -> winio::Result<()> {
    let paths: Vec<PathBuf> = std::env::args_os().skip(1).map(PathBuf::from).collect();
    run_from_paths(paths)
}

pub fn run_from_args(args: &[String]) -> winio::Result<()> {
    run_from_paths(args.iter().map(PathBuf::from).collect())
}

pub fn run_from_args_with_options(args: &[String], options: GuiOptions) -> winio::Result<()> {
    run_from_paths_with_options(args.iter().map(PathBuf::from).collect(), options)
}

pub fn run_from_paths(paths: Vec<PathBuf>) -> winio::Result<()> {
    run_from_paths_with_options(paths, GuiOptions::default())
}

pub fn run_from_paths_with_options(paths: Vec<PathBuf>, options: GuiOptions) -> winio::Result<()> {
    #[cfg(target_os = "windows")]
    if let Some(app_user_model_id) = options.app_user_model_id.as_deref() {
        let _ = set_process_app_user_model_id(app_user_model_id);
    }

    let app_name = options.app_name.clone();
    App::builder()
        .name(app_name.as_str())
        .build()?
        .block_on(MainModel::run((paths, options)))
}

struct LogEntry {
    path: PathBuf,
    response: CambiaResponse,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Mark {
    Normal,
    Good,
    Bad,
    Warn,
    Muted,
    Section,
}

impl Mark {
    /// Text color override for the label, or `None` to keep winio's
    /// theme-aware default (black in light mode, white in dark mode).
    fn color(self, dark: bool) -> Option<u32> {
        let (light, dark_color) = match self {
            Mark::Normal => return None,
            Mark::Good => (rgb(0x15, 0x80, 0x3d), rgb(0x4a, 0xde, 0x80)),
            Mark::Bad => (rgb(0xb9, 0x1c, 0x1c), rgb(0xf8, 0x71, 0x71)),
            Mark::Warn => (rgb(0x92, 0x40, 0x0e), rgb(0xfa, 0xcc, 0x15)),
            Mark::Muted => (rgb(0x6b, 0x72, 0x80), rgb(0x9c, 0xa3, 0xaf)),
            Mark::Section => (rgb(0x1d, 0x4e, 0xd8), rgb(0x93, 0xc5, 0xfd)),
        };
        Some(if dark { dark_color } else { light })
    }
}

const fn rgb(r: u32, g: u32, b: u32) -> u32 {
    r | (g << 8) | (b << 16)
}

#[cfg(target_os = "windows")]
struct LabelFont(HFONT);

#[cfg(target_os = "windows")]
impl Drop for LabelFont {
    fn drop(&mut self) {
        if !self.0.is_null() {
            unsafe {
                DeleteObject(self.0 as HGDIOBJ);
            }
        }
    }
}

#[cfg(feature = "gui_profiler")]
struct GuiProfile {
    reason: &'static str,
    selected: usize,
    parsed: usize,
    rows: usize,
    labels: usize,
    total_start: Instant,
    rebuild_clear_ms: f64,
    build_rows_ms: f64,
    label_init_ms: f64,
    label_text_ms: f64,
    label_color_ms: f64,
    label_font_ms: f64,
    rebuild_total_ms: f64,
    refresh_list_ms: f64,
    refresh_header_ms: f64,
    refresh_internal_ms: f64,
    refresh_tracks_ms: f64,
    refresh_total_ms: f64,
    layout_measure_ms: f64,
    layout_scroll_ms: f64,
    layout_total_ms: f64,
}

#[cfg(feature = "gui_profiler")]
impl GuiProfile {
    fn new(reason: &'static str, selected: usize, parsed: usize) -> Self {
        Self {
            reason,
            selected,
            parsed,
            rows: 0,
            labels: 0,
            total_start: Instant::now(),
            rebuild_clear_ms: 0.0,
            build_rows_ms: 0.0,
            label_init_ms: 0.0,
            label_text_ms: 0.0,
            label_color_ms: 0.0,
            label_font_ms: 0.0,
            rebuild_total_ms: 0.0,
            refresh_list_ms: 0.0,
            refresh_header_ms: 0.0,
            refresh_internal_ms: 0.0,
            refresh_tracks_ms: 0.0,
            refresh_total_ms: 0.0,
            layout_measure_ms: 0.0,
            layout_scroll_ms: 0.0,
            layout_total_ms: 0.0,
        }
    }
}

#[cfg(feature = "gui_profiler")]
macro_rules! profile_time {
    ($slot:expr, $expr:expr) => {{
        let profile_start = Instant::now();
        let profile_result = $expr;
        $slot += elapsed_ms(profile_start);
        profile_result
    }};
}

#[cfg(not(feature = "gui_profiler"))]
macro_rules! profile_time {
    ($slot:expr, $expr:expr) => {{
        $expr
    }};
}

struct DetailRow {
    indent: usize,
    text: String,
    mark: Mark,
    section: bool,
}

impl DetailRow {
    fn section(text: impl Into<String>) -> Self {
        Self {
            indent: 0,
            text: text.into(),
            mark: Mark::Section,
            section: true,
        }
    }

    fn line(indent: usize, text: impl Into<String>, mark: Mark) -> Self {
        Self {
            indent,
            text: text.into(),
            mark,
            section: false,
        }
    }
}

struct MainModel {
    window: Child<Window>,
    file_label: Child<Label>,
    ops_label: Child<Label>,
    log_score_label: Child<Label>,
    encoding_label: Child<Label>,
    loading_label: Child<Label>,
    loading_progress: Child<Progress>,
    open_location_button: Child<Button>,
    list: Child<ListBox>,
    internal_logs: Child<ListBox>,
    tracks: Child<ListBox>,
    details_scroll: Child<ScrollView>,
    details_scroll_spacer: Child<Label>,
    detail_labels: Vec<Child<Label>>,
    detail_rows: Vec<DetailRow>,
    #[cfg(target_os = "windows")]
    detail_fonts: Vec<Option<LabelFont>>,
    #[cfg(target_os = "windows")]
    last_detail_dpi: u32,
    logs: Vec<LogEntry>,
    visible_log_indices: Vec<usize>,
    selected: usize,
    selected_parsed: usize,
    status: String,
    list_visible: bool,
    internal_visible: bool,
    content_dirty: bool,
    detail_scroll_reset_pending: bool,
    list_items_cache: Vec<String>,
    internal_items_cache: Vec<String>,
    track_items_cache: Vec<String>,
    last_client_size: Option<(f64, f64)>,
    loading: bool,
    loading_text: String,
    loading_done: usize,
    loading_total: usize,
    load_generation: u64,
    #[cfg(feature = "gui_profiler")]
    gui_profile: Option<GuiProfile>,
}

enum MainMessage {
    Noop,
    Close,
    Resize,
    SelectLog,
    SelectParsedLog,
    OpenFileLocation,
    LoadPaths(Vec<PathBuf>),
    LoadStarted(u64, String),
    LoadProgress(u64, usize, usize, String),
    LoadFinished(u64, std::result::Result<Vec<LogEntry>, String>),
}

impl Component for MainModel {
    type Error = Error;
    type Event = ();
    type Init<'a> = (Vec<PathBuf>, GuiOptions);
    type Message = MainMessage;

    async fn init(
        init_options: Self::Init<'_>,
        sender: &ComponentSender<Self>,
    ) -> winio::Result<Self> {
        let (init_paths, options) = init_options;
        init! {
            window: Window = (()) => {
                text: "Cambia",
                size: Size::new(1120.0, 760.0),
            },
            file_label: Label = (&window) => {
                text: "文件: N/A",
            },
            ops_label: Label = (&window) => {
                text: "OPS 总分: N/A",
            },
            log_score_label: Label = (&window) => {
                text: "当前内部 log 分数: N/A",
            },
            encoding_label: Label = (&window) => {
                text: "编码: N/A | 内部日志数: 0 | 评估器数: 0",
            },
            loading_label: Label = (&window) => {
                text: "",
            },
            loading_progress: Progress = (&window),
            list: ListBox = (&window),
            internal_logs: ListBox = (&window),
            tracks: ListBox = (&window),
            details_scroll: ScrollView = (&window),
            details_scroll_spacer: Label = (&details_scroll) => {
                text: "",
            },
            open_location_button: Button = (&details_scroll) => {
                text: "打开文件位置",
            },
        }

        window.set_text(options.app_name.as_str())?;
        #[cfg(target_os = "windows")]
        if let Some(icon_resource_id) = options.icon_resource_id {
            window.set_icon_by_id(icon_resource_id).ok();
        }
        #[cfg(target_os = "windows")]
        let h = window.as_window().as_win32();
        #[cfg(target_os = "windows")]
        if let Some(icon_resource_id) = options.icon_resource_id {
            let _ = set_window_icons(h, icon_resource_id);
        }
        #[cfg(target_os = "windows")]
        if let Some(app_user_model_id) = options.app_user_model_id.as_deref() {
            let _ = set_window_app_user_model_properties(
                h,
                app_user_model_id,
                options.app_name.as_str(),
                options.icon_resource_id,
            );
        }

        window.show()?;

        #[cfg(target_os = "windows")]
        install_window_handler(&window, sender.clone());
        #[cfg(target_os = "windows")]
        install_color_parent_handler(&details_scroll);

        let mut model = Self {
            window,
            file_label,
            ops_label,
            log_score_label,
            encoding_label,
            loading_label,
            loading_progress,
            open_location_button,
            list,
            internal_logs,
            tracks,
            details_scroll,
            details_scroll_spacer,
            detail_labels: Vec::new(),
            detail_rows: Vec::new(),
            #[cfg(target_os = "windows")]
            detail_fonts: Vec::new(),
            #[cfg(target_os = "windows")]
            last_detail_dpi: 0,
            logs: Vec::new(),
            visible_log_indices: Vec::new(),
            selected: 0,
            selected_parsed: 0,
            status: String::new(),
            list_visible: false,
            internal_visible: false,
            content_dirty: true,
            detail_scroll_reset_pending: false,
            list_items_cache: Vec::new(),
            internal_items_cache: Vec::new(),
            track_items_cache: Vec::new(),
            last_client_size: None,
            loading: false,
            loading_text: String::new(),
            loading_done: 0,
            loading_total: 0,
            load_generation: 0,
            #[cfg(feature = "gui_profiler")]
            gui_profile: None,
        };

        model.details_scroll.set_hscroll(true)?;
        model.loading_progress.set_visible(false)?;
        model.open_location_button.set_visible(false)?;
        if !init_paths.is_empty() {
            model.start_load(init_paths, sender.clone());
        }
        model.rebuild_detail_labels().await?;
        Ok(model)
    }

    async fn start(&mut self, sender: &ComponentSender<Self>) -> ! {
        start! {
            sender, default: MainMessage::Noop,
            self.window => {
                WindowEvent::Close => MainMessage::Close,
                WindowEvent::Resize => MainMessage::Resize,
            },
            self.open_location_button => {
                ButtonEvent::Click => MainMessage::OpenFileLocation,
            },
            self.list => {
                ListBoxEvent::Select => MainMessage::SelectLog,
            },
            self.internal_logs => {
                ListBoxEvent::Select => MainMessage::SelectParsedLog,
            },
            self.details_scroll => {},
        }
    }

    async fn update_children(&mut self) -> winio::Result<bool> {
        let _ = self.window.update().await?;
        let _ = self.file_label.update().await?;
        let _ = self.ops_label.update().await?;
        let _ = self.log_score_label.update().await?;
        let _ = self.encoding_label.update().await?;
        let _ = self.loading_label.update().await?;
        let _ = self.loading_progress.update().await?;
        let _ = self.open_location_button.update().await?;
        let _ = self.list.update().await?;
        let _ = self.internal_logs.update().await?;
        let _ = self.tracks.update().await?;
        let _ = self.details_scroll.update().await?;
        let _ = self.details_scroll_spacer.update().await?;
        for label in &mut self.detail_labels {
            let _ = label.update().await?;
        }
        Ok(false)
    }

    async fn update(
        &mut self,
        message: Self::Message,
        sender: &ComponentSender<Self>,
    ) -> winio::Result<bool> {
        match message {
            MainMessage::Noop => Ok(false),
            MainMessage::Close => {
                sender.output(());
                Ok(false)
            }
            MainMessage::Resize => {
                let size = self.window.client_size()?;
                let current = (size.width, size.height);
                if self.last_client_size == Some(current) {
                    Ok(false)
                } else {
                    self.last_client_size = Some(current);
                    Ok(true)
                }
            }
            MainMessage::SelectLog => {
                if let Some(index) = selected_index(&self.list, self.visible_log_indices.len()) {
                    let actual_index = self.visible_log_indices[index];
                    if self.selected == actual_index && self.selected_parsed == 0 {
                        return Ok(false);
                    }
                    self.selected = actual_index;
                    self.selected_parsed = 0;
                    #[cfg(feature = "gui_profiler")]
                    self.start_gui_profile("select_log");
                    self.content_dirty = true;
                    self.detail_scroll_reset_pending = true;
                    self.rebuild_detail_labels().await?;
                    return Ok(true);
                }
                Ok(false)
            }
            MainMessage::SelectParsedLog => {
                if let Some(entry) = self.logs.get(self.selected) {
                    if let Some(index) =
                        selected_index(&self.internal_logs, entry.response.parsed.parsed_logs.len())
                    {
                        if self.selected_parsed == index {
                            return Ok(false);
                        }
                        self.selected_parsed = index;
                        #[cfg(feature = "gui_profiler")]
                        self.start_gui_profile("select_parsed_log");
                        self.content_dirty = true;
                        self.detail_scroll_reset_pending = true;
                        self.rebuild_detail_labels().await?;
                        return Ok(true);
                    }
                }
                Ok(false)
            }
            MainMessage::OpenFileLocation => {
                if let Some(entry) = self.current_entry() {
                    let _ = open_file_location(&entry.path);
                }
                Ok(false)
            }
            MainMessage::LoadPaths(paths) => {
                self.start_load(paths, sender.clone());
                Ok(true)
            }
            MainMessage::LoadStarted(generation, text) => {
                if generation != self.load_generation {
                    return Ok(false);
                }
                self.loading = true;
                self.loading_text = text;
                self.loading_done = 0;
                self.loading_total = 0;
                Ok(true)
            }
            MainMessage::LoadProgress(generation, done, total, text) => {
                if generation != self.load_generation {
                    return Ok(false);
                }
                self.loading = true;
                self.loading_done = done;
                self.loading_total = total;
                self.loading_text = text;
                Ok(true)
            }
            MainMessage::LoadFinished(generation, result) => {
                if generation != self.load_generation {
                    return Ok(false);
                }
                self.loading = false;
                self.loading_done = 0;
                self.loading_total = 0;
                self.loading_text.clear();
                self.finish_load(result);
                self.content_dirty = true;
                self.detail_scroll_reset_pending = true;
                self.rebuild_detail_labels().await?;

                #[cfg(target_os = "windows")]
                let h = self.window.as_window().as_win32();
                #[cfg(target_os = "windows")]
                let _ = raise_once(h);
                Ok(true)
            }
        }
    }

    fn render(&mut self, _sender: &ComponentSender<Self>) -> winio::Result<()> {
        let size = self.window.client_size()?;
        self.last_client_size = Some((size.width, size.height));
        let width = size.width.max(620.0);
        let height = size.height.max(420.0);
        let pad = 12.0;
        let list_w = if self.visible_log_indices.len() > 1 {
            320.0
        } else {
            0.0
        };
        self.list_visible = list_w > 0.0;

        let detail_header_y = pad;
        let left_x = pad;
        let detail_x = if self.list_visible {
            pad + list_w + 10.0
        } else {
            pad
        };
        let left_label_w = width - 2.0 * pad;
        let loading_h = if self.loading { ROW_H + 6.0 + 18.0 } else { 0.0 };
        let loading_gap = if self.loading { 10.0 } else { 0.0 };
        let reserved_bottom_h = loading_h + loading_gap;

        self.file_label
            .set_loc(Point::new(left_x, detail_header_y))?;
        self.file_label.set_size(Size::new(left_label_w, ROW_H))?;
        self.ops_label
            .set_loc(Point::new(left_x, detail_header_y + ROW_H + 2.0))?;
        self.ops_label.set_size(Size::new(left_label_w, ROW_H))?;
        self.log_score_label
            .set_loc(Point::new(left_x, detail_header_y + (ROW_H + 2.0) * 2.0))?;
        self.log_score_label
            .set_size(Size::new(left_label_w, ROW_H))?;
        self.encoding_label
            .set_loc(Point::new(left_x, detail_header_y + (ROW_H + 2.0) * 3.0))?;
        self.encoding_label
            .set_size(Size::new(left_label_w, ROW_H))?;
        let loading_y = height - pad - loading_h;
        self.loading_label
            .set_loc(Point::new(pad, loading_y))?;
        self.loading_label
            .set_size(Size::new((width - 2.0 * pad).max(260.0), ROW_H))?;
        self.loading_progress
            .set_loc(Point::new(pad, loading_y + ROW_H + 6.0))?;
        self.loading_progress
            .set_size(Size::new((width - 2.0 * pad).max(260.0), 18.0))?;
        self.loading_label.set_visible(self.loading)?;
        self.loading_progress.set_visible(self.loading)?;
        let has_entry = self.current_entry().is_some();
        self.open_location_button.set_visible(has_entry)?;
        if self.loading_total > 0 {
            self.loading_progress.set_indeterminate(false)?;
            self.loading_progress.set_minimum(0)?;
            self.loading_progress.set_maximum(self.loading_total)?;
            self.loading_progress
                .set_pos(self.loading_done.min(self.loading_total))?;
            self.loading_label.set_text(format!(
                "{} ({}/{})",
                self.loading_text, self.loading_done, self.loading_total
            ))?;
        } else {
            self.loading_progress.set_indeterminate(self.loading)?;
            self.loading_label.set_text(self.loading_text.as_str())?;
        }

        let body_y = detail_header_y + 98.0;

        self.list.set_visible(self.list_visible)?;
        if self.list_visible {
            self.list.set_loc(Point::new(left_x, body_y))?;
            self.list
                .set_size(Size::new(
                    list_w,
                    (height - body_y - pad - reserved_bottom_h).max(120.0),
                ))?;
        } else {
            self.list.set_visible(false)?;
        }

        let content_x = detail_x;
        let content_w = width - content_x - pad;
        let tracks_h = if self.current_entry().is_some() {
            170.0
        } else {
            0.0
        };
        let tracks_y = height - pad - reserved_bottom_h - tracks_h;
        let body_h = (tracks_y - body_y - pad).max(120.0);

        self.internal_logs.set_visible(false)?;
        self.tracks.set_visible(has_entry)?;
        if has_entry {
            self.tracks.set_loc(Point::new(content_x, tracks_y))?;
            self.tracks
                .set_size(Size::new(content_w.max(260.0), tracks_h))?;
        }

        self.details_scroll.set_loc(Point::new(content_x, body_y))?;
        self.details_scroll
            .set_size(Size::new(content_w.max(260.0), body_h))?;

        if self.content_dirty {
            self.refresh_content()?;
            self.content_dirty = false;
        }
        #[cfg(target_os = "windows")]
        self.refresh_detail_fonts_for_dpi();
        self.layout_detail_labels(content_w.max(260.0))?;
        #[cfg(feature = "gui_profiler")]
        self.finish_gui_profile();
        Ok(())
    }

    fn render_children(&mut self) -> winio::Result<()> {
        self.window.render()?;
        self.file_label.render()?;
        self.ops_label.render()?;
        self.log_score_label.render()?;
        self.encoding_label.render()?;
        self.loading_label.render()?;
        self.loading_progress.render()?;
        self.list.render()?;
        self.internal_logs.render()?;
        self.tracks.render()?;
        self.details_scroll.render()?;
        self.details_scroll_spacer.render()?;
        self.open_location_button.render()?;
        for label in &mut self.detail_labels {
            label.render()?;
        }
        Ok(())
    }
}

impl MainModel {
    fn start_load(&mut self, paths: Vec<PathBuf>, sender: ComponentSender<Self>) {
        self.load_generation = self.load_generation.wrapping_add(1);
        let generation = self.load_generation;
        self.loading = true;
        self.loading_text = "正在收集日志文件...".to_string();
        self.loading_done = 0;
        self.loading_total = 0;
        thread::spawn(move || {
            sender.post(MainMessage::LoadStarted(
                generation,
                "正在收集日志文件...".to_string(),
            ));
            let result = collect_log_paths(&paths).and_then(|log_paths| {
                if log_paths.is_empty() {
                    return Ok(Vec::new());
                }
                sender.post(MainMessage::LoadProgress(
                    generation,
                    0,
                    log_paths.len(),
                    "正在解析日志...".to_string(),
                ));
                parse_logs_parallel(log_paths, generation, sender.clone())
            });
            sender.post(MainMessage::LoadFinished(generation, result));
        });
    }

    fn finish_load(&mut self, result: std::result::Result<Vec<LogEntry>, String>) {
        #[cfg(target_os = "windows")]
        clear_colored_labels();
        match result {
            Ok(logs) if logs.is_empty() => {
                self.logs.clear();
                self.visible_log_indices.clear();
                self.selected = 0;
                self.selected_parsed = 0;
                self.status = "未找到任何 .log 文件".to_string();
            }
            Ok(logs) => {
                self.logs = logs;
                self.logs.sort_by(|a, b| a.path.cmp(&b.path));
                self.rebuild_visible_logs();
                self.selected = self.visible_log_indices.first().copied().unwrap_or(0);
                self.selected_parsed = 0;
                self.status = format!("已加载 {} 个日志", self.logs.len());
            }
            Err(err) => {
                self.logs.clear();
                self.visible_log_indices.clear();
                self.selected = 0;
                self.selected_parsed = 0;
                self.status = err;
            }
        }
    }

    fn current_entry(&self) -> Option<&LogEntry> {
        self.logs.get(self.selected)
    }

    fn rebuild_visible_logs(&mut self) {
        self.visible_log_indices = (0..self.logs.len()).collect();
        if self.logs.len() <= 1 {
            return;
        }

        let non_perfect = self
            .visible_log_indices
            .iter()
            .copied()
            .filter(|&index| {
                !is_perfect_score(&ops_score(
                    self.logs[index].response.evaluation_combined.as_slice(),
                ))
            })
            .collect::<Vec<_>>();

        if !non_perfect.is_empty() {
            self.visible_log_indices = non_perfect;
        }
    }

    #[cfg(feature = "gui_profiler")]
    fn start_gui_profile(&mut self, reason: &'static str) {
        self.gui_profile = if gui_profile_enabled() {
            Some(GuiProfile::new(reason, self.selected, self.selected_parsed))
        } else {
            None
        };
    }

    #[cfg(feature = "gui_profiler")]
    fn finish_gui_profile(&mut self) {
        let Some(profile) = self.gui_profile.take() else {
            return;
        };
        write_gui_profile(profile);
    }

    async fn rebuild_detail_labels(&mut self) -> winio::Result<()> {
        #[cfg(feature = "gui_profiler")]
        let total_start = Instant::now();
        #[cfg(feature = "gui_profiler")]
        let clear_start = Instant::now();
        #[cfg(target_os = "windows")]
        clear_colored_labels();
        #[cfg(feature = "gui_profiler")]
        let rebuild_clear_ms = elapsed_ms(clear_start);

        #[cfg(feature = "gui_profiler")]
        let build_start = Instant::now();
        self.detail_rows = if let Some(entry) = self.current_entry() {
            build_detail_rows(entry, self.selected_parsed)
        } else if self.status.is_empty() {
            vec![DetailRow::line(0, EMPTY_TEXT, Mark::Muted)]
        } else {
            vec![
                DetailRow::line(0, self.status.clone(), Mark::Bad),
                DetailRow::line(0, EMPTY_TEXT, Mark::Muted),
            ]
        };
        #[cfg(feature = "gui_profiler")]
        let build_rows_ms = elapsed_ms(build_start);

        #[cfg(feature = "gui_profiler")]
        let mut label_init_ms = 0.0;
        #[cfg(feature = "gui_profiler")]
        let mut label_text_ms = 0.0;
        #[cfg(feature = "gui_profiler")]
        let mut label_color_ms = 0.0;
        #[cfg(feature = "gui_profiler")]
        let mut label_font_ms = 0.0;

        while self.detail_labels.len() < self.detail_rows.len() {
            let label = profile_time!(
                label_init_ms,
                Child::<Label>::init(&self.details_scroll).await?
            );
            self.detail_labels.push(label);
            #[cfg(target_os = "windows")]
            self.detail_fonts.push(None);
        }

        for (idx, row) in self.detail_rows.iter().enumerate() {
            let label = &mut self.detail_labels[idx];

            profile_time!(label_text_ms, {
                label.set_visible(true)?;
                label.set_text(row.text.as_str())?;
            });

            #[cfg(target_os = "windows")]
            {
                profile_time!(label_color_ms, {
                    if row.mark.color(false).is_some() {
                        register_label_color(&label, row.mark);
                    }
                });

                profile_time!(label_font_ms, {
                    if let Some(font) = set_detail_label_font(&label, row) {
                        self.detail_fonts[idx] = Some(font);
                    }
                });
            }
        }

        for label in self.detail_labels.iter_mut().skip(self.detail_rows.len()) {
            profile_time!(label_text_ms, {
                label.set_text("")?;
                label.set_loc(Point::new(0.0, 0.0))?;
                label.set_size(Size::zero())?;
                label.set_visible(false)?;
            });
        }
        #[cfg(feature = "gui_profiler")]
        if let Some(profile) = &mut self.gui_profile {
            profile.rows = self.detail_rows.len();
            profile.labels = self.detail_rows.len();
            profile.rebuild_clear_ms = rebuild_clear_ms;
            profile.build_rows_ms = build_rows_ms;
            profile.label_init_ms = label_init_ms;
            profile.label_text_ms = label_text_ms;
            profile.label_color_ms = label_color_ms;
            profile.label_font_ms = label_font_ms;
            profile.rebuild_total_ms = elapsed_ms(total_start);
        }
        #[cfg(target_os = "windows")]
        {
            self.last_detail_dpi = unsafe { GetDpiForWindow(self.window.as_window().as_win32()) };
        }
        Ok(())
    }

    #[cfg(target_os = "windows")]
    fn refresh_detail_fonts_for_dpi(&mut self) {
        let dpi = unsafe { GetDpiForWindow(self.window.as_window().as_win32()) };
        if self.last_detail_dpi == dpi {
            return;
        }
        self.last_detail_dpi = dpi;
        for (idx, row) in self.detail_rows.iter().enumerate() {
            let Some(label) = self.detail_labels.get(idx) else {
                break;
            };
            if let Some(font) = set_detail_label_font(label, row) {
                if idx >= self.detail_fonts.len() {
                    self.detail_fonts.resize_with(idx + 1, || None);
                }
                self.detail_fonts[idx] = Some(font);
            }
        }
    }

    fn refresh_content(&mut self) -> winio::Result<()> {
        #[cfg(feature = "gui_profiler")]
        let total_start = Instant::now();
        #[cfg(feature = "gui_profiler")]
        let list_start = Instant::now();
        let list_items = self
            .visible_log_indices
            .iter()
            .map(|&index| left_text(self.list_visible, list_item_text(&self.logs[index])))
            .collect::<Vec<_>>();
        if self.list_items_cache != list_items {
            self.list.set_items(list_items.clone())?;
            self.list_items_cache = list_items;
        }
        if !self.visible_log_indices.is_empty() {
            if !self.visible_log_indices.contains(&self.selected) {
                self.selected = self.visible_log_indices[0];
            }
            if self.list_visible {
                let visible_index = self
                    .visible_log_indices
                    .iter()
                    .position(|&index| index == self.selected)
                    .unwrap_or(0);
                if !self.list.is_selected(visible_index).unwrap_or(false) {
                    self.list.set_selected(visible_index, true).ok();
                }
            }
        }
        #[cfg(feature = "gui_profiler")]
        let refresh_list_ms = elapsed_ms(list_start);

        #[cfg(feature = "gui_profiler")]
        let refresh_header_ms: f64;
        #[cfg(feature = "gui_profiler")]
        let refresh_internal_ms: f64;
        #[cfg(feature = "gui_profiler")]
        let refresh_tracks_ms: f64;
        if let Some(entry) = self.logs.get(self.selected) {
            let parsed_len = entry.response.parsed.parsed_logs.len();
            self.selected_parsed = self.selected_parsed.min(parsed_len.saturating_sub(1));
            self.internal_visible = parsed_len > 1;

            #[cfg(feature = "gui_profiler")]
            let header_start = Instant::now();
            let file_text = format!("文件: {}", display_name(&entry.path));
            let ops_text = format!(
                "OPS 总分: {}",
                ops_score(entry.response.evaluation_combined.as_slice())
            );
            let log_score_text = format!(
                "当前内部 log 分数: {}",
                current_log_score(entry, self.selected_parsed)
            );
            let encoding_text = format!(
                "编码: {} | 内部日志数: {} | 评估器数: {} | Log ID: {}",
                entry.response.parsed.encoding,
                parsed_len,
                entry.response.evaluation_combined.len(),
                hex::encode(&entry.response.id)
            );
            self.file_label.set_text(file_text)?;
            self.ops_label.set_text(ops_text)?;
            self.log_score_label.set_text(log_score_text)?;
            self.encoding_label.set_text(encoding_text)?;
            #[cfg(feature = "gui_profiler")]
            {
                refresh_header_ms = elapsed_ms(header_start);
            }

            #[cfg(feature = "gui_profiler")]
            let internal_start = Instant::now();
            let internal_items = entry
                .response
                .parsed
                .parsed_logs
                .iter()
                .enumerate()
                .map(|(idx, log)| {
                    format!(
                        "Log #{} | {} | 校验: {}",
                        idx + 1,
                        log.drive,
                        integrity_text(&log.checksum.integrity)
                    )
                })
                .collect::<Vec<_>>();
            if self.internal_items_cache != internal_items {
                self.internal_logs.set_items(internal_items.clone())?;
                self.internal_items_cache = internal_items;
            }
            if self.internal_visible
                && !self
                    .internal_logs
                    .is_selected(self.selected_parsed)
                    .unwrap_or(false)
            {
                self.internal_logs
                    .set_selected(self.selected_parsed, true)
                    .ok();
            }
            #[cfg(feature = "gui_profiler")]
            {
                refresh_internal_ms = elapsed_ms(internal_start);
            }

            #[cfg(feature = "gui_profiler")]
            let tracks_start = Instant::now();
            let track_items = track_list_items(entry, self.selected_parsed);
            if self.track_items_cache != track_items {
                self.tracks.set_items(track_items.clone())?;
                self.track_items_cache = track_items;
            }
            #[cfg(feature = "gui_profiler")]
            {
                refresh_tracks_ms = elapsed_ms(tracks_start);
            }
        } else {
            #[cfg(feature = "gui_profiler")]
            let header_start = Instant::now();
            let status = if self.status.is_empty() {
                EMPTY_TEXT.to_string()
            } else {
                format!("{} {}", self.status, EMPTY_TEXT)
            };
            self.internal_visible = false;
            self.file_label.set_text("文件: N/A")?;
            self.ops_label.set_text("OPS 总分: N/A")?;
            self.log_score_label.set_text("当前内部 log 分数: N/A")?;
            self.encoding_label
                .set_text("编码: N/A | 内部日志数: 0 | 评估器数: 0")?;
            #[cfg(feature = "gui_profiler")]
            {
                refresh_header_ms = elapsed_ms(header_start);
            }

            #[cfg(feature = "gui_profiler")]
            let internal_start = Instant::now();
            let internal_items = Vec::<String>::new();
            if self.internal_items_cache != internal_items {
                self.internal_logs.set_items(internal_items.clone())?;
                self.internal_items_cache = internal_items;
            }
            #[cfg(feature = "gui_profiler")]
            {
                refresh_internal_ms = elapsed_ms(internal_start);
            }

            #[cfg(feature = "gui_profiler")]
            let tracks_start = Instant::now();
            let track_items = vec![status];
            if self.track_items_cache != track_items {
                self.tracks.set_items(track_items.clone())?;
                self.track_items_cache = track_items;
            }
            #[cfg(feature = "gui_profiler")]
            {
                refresh_tracks_ms = elapsed_ms(tracks_start);
            }
        }

        #[cfg(feature = "gui_profiler")]
        if let Some(profile) = &mut self.gui_profile {
            profile.refresh_list_ms = refresh_list_ms;
            profile.refresh_header_ms = refresh_header_ms;
            profile.refresh_internal_ms = refresh_internal_ms;
            profile.refresh_tracks_ms = refresh_tracks_ms;
            profile.refresh_total_ms = elapsed_ms(total_start);
        }
        Ok(())
    }

    fn layout_detail_labels(&mut self, width: f64) -> winio::Result<()> {
        #[cfg(feature = "gui_profiler")]
        let total_start = Instant::now();
        #[cfg(feature = "gui_profiler")]
        let mut measure_ms = 0.0;
        let (right_pad, bottom_pad) = scrollbar_padding(&self.details_scroll);
        let mut y = DETAIL_PAD;
        let mut max_label_right = DETAIL_PAD;
        for (idx, label) in self
            .detail_labels
            .iter_mut()
            .take(self.detail_rows.len())
            .enumerate()
        {
            let row = &self.detail_rows[idx];
            let h = detail_row_height(row);
            let x = DETAIL_PAD + row.indent as f64 * 18.0;
            if idx == OVERVIEW_ROW_COUNT {
                y += OPEN_BUTTON_H + OPEN_BUTTON_GAP;
            }
            label.set_loc(Point::new(x, y))?;
            let preferred = profile_time!(measure_ms, label.preferred_size()?);
            let visible_w = (width - x - DETAIL_PAD - right_pad).max(120.0);
            let label_w = visible_w.max(120.0).max(preferred.width + 12.0);
            label.set_size(Size::new(label_w, h))?;
            max_label_right = max_label_right.max(x + label_w);
            y += h;
        }
        let button_y = DETAIL_PAD
            + self
                .detail_rows
                .iter()
                .take(OVERVIEW_ROW_COUNT)
                .map(detail_row_height)
                .sum::<f64>()
            + 4.0;
        self.open_location_button
            .set_loc(Point::new(DETAIL_PAD + 18.0, button_y))?;
        self.open_location_button
            .set_size(Size::new(132.0, OPEN_BUTTON_H))?;
        let spacer_x = max_label_right.max(width - DETAIL_PAD - right_pad);
        let spacer_y = y + bottom_pad;
        self.details_scroll_spacer
            .set_loc(Point::new(spacer_x, spacer_y))?;
        self.details_scroll_spacer
            .set_size(Size::new(right_pad, bottom_pad))?;
        self.details_scroll_spacer.set_visible(true)?;
        #[cfg(feature = "gui_profiler")]
        let scroll_start = Instant::now();
        self.details_scroll.set_size(self.details_scroll.size()?)?;
        if self.detail_scroll_reset_pending {
            #[cfg(target_os = "windows")]
            reset_details_scroll_position(&self.details_scroll);
            self.detail_scroll_reset_pending = false;
        }
        #[cfg(feature = "gui_profiler")]
        let layout_scroll_ms = elapsed_ms(scroll_start);
        #[cfg(feature = "gui_profiler")]
        if let Some(profile) = &mut self.gui_profile {
            profile.layout_measure_ms = measure_ms;
            profile.layout_scroll_ms = layout_scroll_ms;
            profile.layout_total_ms = elapsed_ms(total_start);
        }
        Ok(())
    }
}

fn detail_row_height(row: &DetailRow) -> f64 {
    if row.section {
        SECTION_H
    } else if row.indent == 0 {
        24.0
    } else if row.indent >= 3 {
        20.0
    } else {
        ROW_H
    }
}

fn scrollbar_padding(scroll: &ScrollView) -> (f64, f64) {
    #[cfg(target_os = "windows")]
    {
        let dpi = unsafe { GetDpiForWindow(scroll.as_widget().as_win32()) };
        let right =
            unsafe { GetSystemMetricsForDpi(SM_CXVSCROLL, dpi) } as f64 + SCROLLBAR_PAD_EXTRA;
        let bottom =
            unsafe { GetSystemMetricsForDpi(SM_CYHSCROLL, dpi) } as f64 + SCROLLBAR_PAD_EXTRA;
        (
            right.max(SCROLLBAR_PAD_EXTRA),
            bottom.max(SCROLLBAR_PAD_EXTRA),
        )
    }
    #[cfg(not(target_os = "windows"))]
    {
        let _ = scroll;
        let fallback = 18.0 + SCROLLBAR_PAD_EXTRA;
        (fallback, fallback)
    }
}

#[cfg(target_os = "windows")]
fn set_detail_label_font(label: &Label, row: &DetailRow) -> Option<LabelFont> {
    let hwnd = label.as_widget().as_win32();
    unsafe {
        let mut metrics: NONCLIENTMETRICSW = std::mem::zeroed();
        metrics.cbSize = std::mem::size_of::<NONCLIENTMETRICSW>() as u32;
        if SystemParametersInfoForDpi(
            SPI_GETNONCLIENTMETRICS,
            metrics.cbSize,
            (&mut metrics as *mut NONCLIENTMETRICSW).cast(),
            0,
            GetDpiForWindow(hwnd),
        ) == 0
        {
            return None;
        }

        let mut font: LOGFONTW = metrics.lfMessageFont;
        let (scale, weight) = detail_font_style(row, font.lfWeight);
        font.lfHeight = scale_logical_font_height(font.lfHeight, scale);
        font.lfWeight = weight;

        let created = CreateFontIndirectW(&font);
        if created.is_null() {
            return None;
        }

        SendMessageW(hwnd, WM_SETFONT, created as usize, 1);
        Some(LabelFont(created))
    }
}

#[cfg(target_os = "windows")]
fn detail_font_style(row: &DetailRow, current_weight: i32) -> (f64, i32) {
    if row.section {
        (1.18, FW_BOLD as i32)
    } else if row.indent == 0 {
        (1.08, current_weight.max(600))
    } else if row.indent >= 3 {
        (0.94, current_weight)
    } else {
        (1.0, current_weight)
    }
}

#[cfg(target_os = "windows")]
fn scale_logical_font_height(height: i32, scale: f64) -> i32 {
    let scaled = (height as f64 * scale).round() as i32;
    if scaled == 0 {
        height.signum()
    } else {
        scaled
    }
}

#[cfg(target_os = "windows")]
fn reset_details_scroll_position(scroll: &ScrollView) {
    let handle = scroll.as_widget().as_win32();
    let view = scroll.as_container().as_win32();
    unsafe {
        let info = SCROLLINFO {
            cbSize: std::mem::size_of::<SCROLLINFO>() as u32,
            fMask: SIF_POS,
            nMin: 0,
            nMax: 0,
            nPage: 0,
            nPos: 0,
            nTrackPos: 0,
        };
        SetScrollInfo(handle, SB_VERT, &info, 1);
        SetScrollInfo(handle, SB_HORZ, &info, 1);
        SetWindowPos(
            view,
            std::ptr::null_mut(),
            0,
            0,
            0,
            0,
            SWP_NOSIZE | SWP_NOZORDER | SWP_NOACTIVATE,
        );
    }
}

#[cfg(feature = "gui_profiler")]
fn elapsed_ms(start: Instant) -> f64 {
    start.elapsed().as_secs_f64() * 1000.0
}

#[cfg(feature = "gui_profiler")]
fn gui_profile_enabled() -> bool {
    std::env::var_os("CAMBIA_GUI_PROFILE")
        .and_then(|value| value.into_string().ok())
        .map(|value| {
            let value = value.trim().to_ascii_lowercase();
            !(value.is_empty() || value == "0" || value == "false" || value == "off")
        })
        .unwrap_or(false)
}

#[cfg(feature = "gui_profiler")]
fn write_gui_profile(profile: GuiProfile) {
    let path = std::env::var_os("CAMBIA_GUI_PROFILE_PATH")
        .map(PathBuf::from)
        .unwrap_or_else(|| std::env::temp_dir().join("cambia-gui-profile.log"));
    let Ok(mut file) = fs::OpenOptions::new().create(true).append(true).open(path) else {
        return;
    };
    let _ = writeln!(
        file,
        "ts={} reason={} selected={} parsed={} rows={} labels={} total_ms={:.3} \
         rebuild_total_ms={:.3} rebuild_clear_ms={:.3} build_rows_ms={:.3} \
         label_init_ms={:.3} label_text_ms={:.3} label_color_ms={:.3} label_font_ms={:.3} \
         refresh_total_ms={:.3} refresh_list_ms={:.3} refresh_header_ms={:.3} \
         refresh_internal_ms={:.3} refresh_tracks_ms={:.3} \
         layout_total_ms={:.3} layout_measure_ms={:.3} layout_scroll_ms={:.3}",
        profile_timestamp_ms(),
        profile.reason,
        profile.selected,
        profile.parsed,
        profile.rows,
        profile.labels,
        elapsed_ms(profile.total_start),
        profile.rebuild_total_ms,
        profile.rebuild_clear_ms,
        profile.build_rows_ms,
        profile.label_init_ms,
        profile.label_text_ms,
        profile.label_color_ms,
        profile.label_font_ms,
        profile.refresh_total_ms,
        profile.refresh_list_ms,
        profile.refresh_header_ms,
        profile.refresh_internal_ms,
        profile.refresh_tracks_ms,
        profile.layout_total_ms,
        profile.layout_measure_ms,
        profile.layout_scroll_ms,
    );
}

#[cfg(feature = "gui_profiler")]
fn profile_timestamp_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_millis())
        .unwrap_or_default()
}

fn selected_index(list: &ListBox, len: usize) -> Option<usize> {
    #[cfg(target_os = "windows")]
    {
        let index = unsafe { SendMessageW(list.as_widget().as_win32(), LB_GETCURSEL, 0, 0) };
        if index == LB_ERR as isize {
            None
        } else {
            let index = index as usize;
            (index < len).then_some(index)
        }
    }
    #[cfg(not(target_os = "windows"))]
    {
        (0..len).find(|&index| list.is_selected(index).unwrap_or(false))
    }
}

fn collect_log_paths(inputs: &[PathBuf]) -> std::result::Result<Vec<PathBuf>, String> {
    let mut logs = Vec::new();
    for input in inputs {
        let metadata = fs::metadata(input)
            .map_err(|err| format!("无法访问路径 {}: {err}", input.display()))?;
        if metadata.is_dir() {
            for entry in WalkDir::new(input)
                .into_iter()
                .filter_map(|entry| entry.ok())
            {
                let path = entry.path();
                if entry.file_type().is_file() && has_log_extension(path) {
                    logs.push(path.to_path_buf());
                }
            }
        } else if metadata.is_file() {
            if has_log_extension(input) {
                logs.push(input.clone());
            }
        } else {
            return Err(format!("路径 {} 既不是文件也不是目录", input.display()));
        }
    }
    logs.sort();
    logs.dedup();
    Ok(logs)
}

fn parse_logs_parallel(
    paths: Vec<PathBuf>,
    generation: u64,
    sender: ComponentSender<MainModel>,
) -> std::result::Result<Vec<LogEntry>, String> {
    let total = paths.len();
    if total == 0 {
        return Ok(Vec::new());
    }

    let worker_count = thread::available_parallelism()
        .map(|value| value.get())
        .unwrap_or(4)
        .min(total)
        .max(1);
    let (task_tx, task_rx) = mpsc::channel::<PathBuf>();
    let (result_tx, result_rx) = mpsc::channel::<std::result::Result<LogEntry, String>>();
    let shared_rx = Arc::new(Mutex::new(task_rx));

    for _ in 0..worker_count {
        let rx = Arc::clone(&shared_rx);
        let tx = result_tx.clone();
        thread::spawn(move || loop {
            let path = {
                let receiver = rx.lock().unwrap();
                match receiver.recv() {
                    Ok(path) => path,
                    Err(_) => break,
                }
            };
            let result = parse_single_log(&path).map(|response| LogEntry { path, response });
            let _ = tx.send(result);
        });
    }
    drop(result_tx);

    for path in paths {
        let _ = task_tx.send(path);
    }
    drop(task_tx);

    let mut logs = Vec::new();
    let mut errors = Vec::new();
    for done in 1..=total {
        match result_rx.recv() {
            Ok(Ok(entry)) => logs.push(entry),
            Ok(Err(err)) => errors.push(err),
            Err(_) => break,
        }
        sender.post(MainMessage::LoadProgress(
            generation,
            done,
            total,
            format!("正在解析日志: {}", done),
        ));
    }

    if logs.is_empty() && !errors.is_empty() {
        Err(errors.join("\r\n"))
    } else {
        Ok(logs)
    }
}

fn parse_single_log(path: &Path) -> std::result::Result<CambiaResponse, String> {
    let bytes = fs::read(path).map_err(|err| format!("无法读取文件 {}: {err}", path.display()))?;
    parse_log_bytes(Vec::new(), &bytes)
        .map_err(|err| format!("解析日志失败 {}: {err}", path.display()))
}

fn has_log_extension(path: &Path) -> bool {
    path.extension()
        .and_then(|ext| ext.to_str())
        .map(|ext| ext.eq_ignore_ascii_case("log"))
        .unwrap_or(false)
}

fn list_item_text(entry: &LogEntry) -> String {
    format!(
        "{} | 驱动器: {} | OPS: {} | 校验: {}",
        entry
            .path
            .file_name()
            .and_then(|name| name.to_str())
            .unwrap_or(""),
        drive_text(entry, 0),
        ops_score(entry.response.evaluation_combined.as_slice()),
        checksum_text(entry, 0),
    )
}

fn left_text(enabled: bool, text: String) -> String {
    if enabled {
        truncate_middle(&text, LEFT_TEXT_CHARS)
    } else {
        text
    }
}

fn display_name(path: &Path) -> String {
    match path.file_name().and_then(|name| name.to_str()) {
        Some(name) => name.to_string(),
        None => path.to_string_lossy().to_string(),
    }
}

fn truncate_middle(text: &str, max_chars: usize) -> String {
    let char_count = text.chars().count();
    if char_count <= max_chars {
        return text.to_string();
    }

    let marker = "...";
    let marker_chars = marker.chars().count();
    if max_chars <= marker_chars {
        return marker.chars().take(max_chars).collect();
    }

    let keep = max_chars - marker_chars;
    let head_len = keep / 2;
    let tail_len = keep - head_len;
    let head = text.chars().take(head_len).collect::<String>();
    let tail = text
        .chars()
        .rev()
        .take(tail_len)
        .collect::<Vec<_>>()
        .into_iter()
        .rev()
        .collect::<String>();
    format!("{head}{marker}{tail}")
}

fn drive_text(entry: &LogEntry, parsed_index: usize) -> String {
    entry
        .response
        .parsed
        .parsed_logs
        .get(parsed_index)
        .or_else(|| entry.response.parsed.parsed_logs.first())
        .map(|log| log.drive.as_str())
        .filter(|drive| !drive.is_empty())
        .unwrap_or("未知")
        .to_string()
}

fn checksum_text(entry: &LogEntry, parsed_index: usize) -> &'static str {
    match entry
        .response
        .parsed
        .parsed_logs
        .get(parsed_index)
        .or_else(|| entry.response.parsed.parsed_logs.first())
        .map(|log| &log.checksum.integrity)
    {
        Some(Integrity::Match) => "匹配",
        Some(Integrity::Mismatch) => "不匹配",
        _ => "未知",
    }
}

fn integrity_text(integrity: &Integrity) -> &'static str {
    match integrity {
        Integrity::Match => "匹配",
        Integrity::Mismatch => "不匹配",
        Integrity::Unknown => "未知",
    }
}

fn integrity_mark(integrity: &Integrity) -> Mark {
    match integrity {
        Integrity::Match => Mark::Good,
        Integrity::Mismatch => Mark::Bad,
        Integrity::Unknown => Mark::Warn,
    }
}

fn ripper_text(value: &Ripper) -> &'static str {
    match value {
        Ripper::EAC => "Exact Audio Copy",
        Ripper::XLD => "X Lossless Decoder",
        Ripper::Whipper => "Whipper",
        Ripper::CueRipper => "CUERipper",
        Ripper::DBPA => "dBpoweramp",
        Ripper::CyanRip => "cyanrip",
        Ripper::EZCD => "EZ CD Audio Converter",
        Ripper::Morituri => "morituri",
        Ripper::Rip => "Rip",
        Ripper::FreAc => "fre:ac",
        Ripper::Other => "Other",
    }
}

fn media_type_text(value: &MediaType) -> &'static str {
    match value {
        MediaType::Pressed => "Pressed",
        MediaType::CDR => "CD-R",
        MediaType::Other => "Other",
        MediaType::Unknown => "Unknown",
    }
}

fn quartet_text(value: &Quartet) -> &'static str {
    match value {
        Quartet::True => "True",
        Quartet::False => "False",
        Quartet::Unknown => "Unknown",
        Quartet::Unsupported => "Unsupported",
    }
}

fn read_mode_text(value: &ReadMode) -> &'static str {
    match value {
        ReadMode::Secure => "Secure",
        ReadMode::Paranoid => "Paranoid",
        ReadMode::Fast => "Fast",
        ReadMode::Burst => "Burst",
        ReadMode::Unknown => "Unknown",
    }
}

fn gap_text(value: &Gap) -> &'static str {
    match value {
        Gap::Append => "Append",
        Gap::AppendNoHtoa => "Append (except HTOA)",
        Gap::AppendUndetected => "Append (gaps undetected)",
        Gap::Prepend => "Prepend",
        Gap::Discard => "Discard",
        Gap::Unknown => "Unknown",
        Gap::Inapplicable => "Inapplicable",
    }
}

fn ar_status_text(value: &AccurateRipStatus) -> &'static str {
    match value {
        AccurateRipStatus::Match => "Match",
        AccurateRipStatus::Mismatch => "Mismatch",
        AccurateRipStatus::Offsetted => "Offsetted",
        AccurateRipStatus::NotFound => "NotFound",
        AccurateRipStatus::Disabled => "Disabled",
    }
}

fn confidence_total_text(value: &AccurateRipConfidenceTotal) -> String {
    match value {
        AccurateRipConfidenceTotal::All(total) => format!("All({total})"),
        AccurateRipConfidenceTotal::Version(total) => format!("Version({total})"),
    }
}

fn time_text(value: &Time) -> String {
    serde_json::to_string(value)
        .map(|text| text.trim_matches('"').to_string())
        .unwrap_or_else(|_| "-".to_string())
}

fn ops_combined(response: &CambiaResponse) -> Option<&EvaluationCombined> {
    response
        .evaluation_combined
        .iter()
        .find(|evaluation| matches!(evaluation.evaluator, EvaluatorType::OPS))
}

fn ops_score(evaluations: &[EvaluationCombined]) -> String {
    evaluations
        .iter()
        .find(|evaluation| matches!(evaluation.evaluator, EvaluatorType::OPS))
        .map(|evaluation| evaluation.combined_score.clone())
        .unwrap_or_else(|| "N/A".to_string())
}

fn is_perfect_score(score: &str) -> bool {
    score.trim().parse::<i32>().ok() == Some(100)
}

fn current_log_score(entry: &LogEntry, parsed_index: usize) -> String {
    ops_combined(&entry.response)
        .and_then(|ops| ops.evaluations.get(parsed_index))
        .map(|evaluation| evaluation.score.clone())
        .unwrap_or_else(|| "N/A".to_string())
}

fn build_detail_rows(entry: &LogEntry, parsed_index: usize) -> Vec<DetailRow> {
    let mut rows = Vec::new();
    let Some(log) = entry.response.parsed.parsed_logs.get(parsed_index) else {
        rows.push(DetailRow::line(0, "当前内部 log 不存在", Mark::Bad));
        return rows;
    };
    let ops = ops_combined(&entry.response);
    let evaluation = ops.and_then(|ops| ops.evaluations.get(parsed_index));

    rows.push(DetailRow::section("概览"));
    rows.push(DetailRow::line(
        1,
        format!("文件: {}", display_name(&entry.path)),
        Mark::Normal,
    ));
    rows.push(DetailRow::line(
        1,
        format!("Log ID: {}", hex::encode(&entry.response.id)),
        Mark::Normal,
    ));
    rows.push(DetailRow::line(
        1,
        format!("编码: {}", entry.response.parsed.encoding),
        Mark::Normal,
    ));
    rows.push(DetailRow::line(
        1,
        format!(
            "内部 log: {}/{}",
            parsed_index + 1,
            entry.response.parsed.parsed_logs.len()
        ),
        Mark::Normal,
    ));
    rows.push(DetailRow::line(
        1,
        format!(
            "OPS 总分: {}",
            ops.map(|ops| ops.combined_score.as_str()).unwrap_or("N/A")
        ),
        score_mark(ops.map(|ops| ops.combined_score.as_str()).unwrap_or("")),
    ));
    rows.push(DetailRow::line(
        1,
        format!(
            "当前内部 log 分数: {}",
            evaluation.map(|ev| ev.score.as_str()).unwrap_or("N/A")
        ),
        score_mark(evaluation.map(|ev| ev.score.as_str()).unwrap_or("")),
    ));

    rows.push(DetailRow::section("Release"));
    rows.push(DetailRow::line(
        1,
        format!("Artist: {}", log.release_info.artist),
        Mark::Normal,
    ));
    rows.push(DetailRow::line(
        1,
        format!("Title: {}", log.release_info.title),
        Mark::Normal,
    ));
    rows.push(DetailRow::line(
        1,
        format!("MusicBrainz TOC: {}", empty_dash(&log.toc.mbz.hash)),
        Mark::Normal,
    ));

    rows.push(DetailRow::section("Evaluations"));
    if entry.response.evaluation_combined.is_empty() {
        rows.push(DetailRow::line(1, "没有评估器结果", Mark::Warn));
    }
    for combined in &entry.response.evaluation_combined {
        rows.push(DetailRow::line(
            1,
            format!(
                "{:?} | combined_score: {} | logs: {}",
                combined.evaluator,
                combined.combined_score,
                combined.evaluations.len()
            ),
            score_mark(&combined.combined_score),
        ));
        if let Some(ev) = combined.evaluations.get(parsed_index) {
            rows.push(DetailRow::line(
                2,
                format!("score: {}", ev.score),
                score_mark(&ev.score),
            ));
            append_evaluation_units(&mut rows, ev);
        }
    }

    rows.push(DetailRow::section("Rip Info"));
    append_checked_value(
        &mut rows,
        "Ripper",
        format!("{} {}", ripper_text(&log.ripper), log.ripper_version),
        evaluation,
        &[
            EvaluationUnitField::Ripper,
            EvaluationUnitField::RipperVersion,
        ],
    );
    append_checked_value(
        &mut rows,
        "Read mode",
        read_mode_text(&log.read_mode),
        evaluation,
        &[EvaluationUnitField::ReadMode],
    );
    append_plain_value(
        &mut rows,
        "Combined R/W offset",
        format_optional(log.combined_rw_offset),
    );
    append_plain_value(&mut rows, "Language", log.language.clone());
    append_checked_value(
        &mut rows,
        "Gap handling",
        gap_text(&log.gap_handling),
        evaluation,
        &[EvaluationUnitField::Gap],
    );
    append_checked_value(
        &mut rows,
        "Drive",
        log.drive.clone(),
        evaluation,
        &[EvaluationUnitField::Drive],
    );
    append_checked_value(
        &mut rows,
        "Read offset",
        format_optional(log.read_offset),
        evaluation,
        &[EvaluationUnitField::Offset],
    );
    append_plain_value(&mut rows, "Media type", media_type_text(&log.media_type));
    append_plain_value(
        &mut rows,
        "Audio encoder",
        if log.audio_encoder.is_empty() {
            "-".to_string()
        } else {
            log.audio_encoder.join(", ")
        },
    );

    rows.push(DetailRow::section("Rip Settings"));
    append_checked_value(
        &mut rows,
        "Accurate stream",
        quartet_text(&log.accurate_stream),
        evaluation,
        &[EvaluationUnitField::AccurateStream],
    );
    append_checked_value(
        &mut rows,
        "Defeat audio cache",
        quartet_text(&log.defeat_audio_cache),
        evaluation,
        &[EvaluationUnitField::Cache],
    );
    append_checked_value(
        &mut rows,
        "Use C2",
        quartet_text(&log.use_c2),
        evaluation,
        &[EvaluationUnitField::C2],
    );
    append_plain_value(&mut rows, "Overread", quartet_text(&log.overread));
    append_checked_value(
        &mut rows,
        "Fill missing samples",
        quartet_text(&log.fill_silence),
        evaluation,
        &[EvaluationUnitField::Samples],
    );
    append_checked_value(
        &mut rows,
        "Delete leading/trailing silence",
        quartet_text(&log.delete_silence),
        evaluation,
        &[EvaluationUnitField::SilentBlocks],
    );
    append_checked_value(
        &mut rows,
        "Null samples in CRC",
        quartet_text(&log.use_null_samples),
        evaluation,
        &[EvaluationUnitField::NullSamples],
    );
    append_checked_value(
        &mut rows,
        "Test and copy",
        quartet_text(&log.test_and_copy),
        evaluation,
        &[EvaluationUnitField::TestAndCopy],
    );
    append_checked_value(
        &mut rows,
        "Normalize",
        quartet_text(&log.normalize),
        evaluation,
        &[EvaluationUnitField::Normalization],
    );
    append_checked_value(
        &mut rows,
        "Add ID3",
        quartet_text(&log.id3_enabled),
        evaluation,
        &[EvaluationUnitField::Tag],
    );

    rows.push(DetailRow::section("Checksum"));
    rows.push(DetailRow::line(
        1,
        format!("Integrity: {}", integrity_text(&log.checksum.integrity)),
        integrity_mark(&log.checksum.integrity),
    ));
    append_plain_value(&mut rows, "Log checksum", empty_dash(&log.checksum.log));
    append_plain_value(
        &mut rows,
        "Calculated checksum",
        empty_dash(&log.checksum.calculated),
    );

    rows.push(DetailRow::section("Tracks"));
    rows.push(DetailRow::line(1, release_line(log), Mark::Section));
    if log.tracks.is_empty() {
        rows.push(DetailRow::line(1, "没有 track 数据", Mark::Warn));
    } else {
        for track in &log.tracks {
            let mark = track_mark(track);
            rows.push(DetailRow::line(1, track_summary(track), mark));
        }
        for track in &log.tracks {
            append_track_details(&mut rows, track);
        }
    }

    rows.push(DetailRow::section("AccurateRip"));
    if log.tracks.is_empty() {
        rows.push(DetailRow::line(1, "没有 AccurateRip 数据", Mark::Warn));
    }
    for track in &log.tracks {
        if let Some(ar) = track.ar_info.last() {
            rows.push(DetailRow::line(
                1,
                format!(
                    "Track {} | {} | sign: {} | offset_sign: {}",
                    track.num,
                    ar_status_text(&ar.status),
                    empty_dash(&ar.sign),
                    empty_dash(&ar.offset_sign)
                ),
                ar_status_mark(&ar.status),
            ));
            if let Some(confidence) = &ar.confidence {
                rows.push(DetailRow::line(
                    2,
                    format!(
                        "confidence matching: {} | total: {} | offset: {}",
                        confidence
                            .matching
                            .map(|v| v.to_string())
                            .unwrap_or_else(|| "-".to_string()),
                        confidence
                            .total
                            .as_ref()
                            .map(confidence_total_text)
                            .unwrap_or_else(|| "-".to_string()),
                        format_ar_offset(&confidence.offset),
                    ),
                    ar_status_mark(&ar.status),
                ));
            }
        } else {
            rows.push(DetailRow::line(
                1,
                format!("Track {} | No AR Data", track.num),
                Mark::Warn,
            ));
        }
    }
    rows
}

fn append_evaluation_units(rows: &mut Vec<DetailRow>, evaluation: &Evaluation) {
    let mut groups: BTreeMap<String, Vec<_>> = BTreeMap::new();
    for unit in &evaluation.evaluation_units {
        groups
            .entry(format_scope(&unit.data.scope))
            .or_default()
            .push(unit);
    }
    if groups.is_empty() {
        return;
    }
    for (scope, units) in groups {
        rows.push(DetailRow::line(2, scope, Mark::Section));
        for unit in units {
            rows.push(DetailRow::line(
                3,
                format!(
                    "field: {:?} | class: {:?} | unit_score: {} | message: {}",
                    unit.data.field, unit.data.class, unit.unit_score, unit.data.message
                ),
                unit_mark(&unit.data.class, &unit.unit_score),
            ));
        }
    }
}

fn append_track_details(rows: &mut Vec<DetailRow>, track: &TrackEntry) {
    rows.push(DetailRow::line(
        1,
        format!("当前 Track {}", track.num),
        Mark::Section,
    ));
    rows.push(DetailRow::line(
        2,
        format!(
            "Track splitting: {}",
            if track.is_range { "Range" } else { "Split" }
        ),
        Mark::Normal,
    ));
    rows.push(DetailRow::line(
        2,
        format!("Aborted: {}", track.aborted),
        if track.aborted { Mark::Bad } else { Mark::Good },
    ));
    append_plain_value(
        rows,
        "Filenames",
        if track.filenames.is_empty() {
            "-".to_string()
        } else {
            track.filenames.join(" | ")
        },
    );
    append_plain_value(
        rows,
        "Peak level",
        track
            .peak_level
            .map(|v| format!("{v:.3}"))
            .unwrap_or_else(|| "-".to_string()),
    );
    append_plain_value(
        rows,
        "Pregap length",
        track
            .pregap_length
            .map(|v| time_text(&v))
            .unwrap_or_else(|| "-".to_string()),
    );
    append_plain_value(
        rows,
        "Extraction speed",
        track
            .extraction_speed
            .map(|v| format!("{v:.1}x"))
            .unwrap_or_else(|| "-".to_string()),
    );
    append_plain_value(
        rows,
        "Gain",
        track
            .gain
            .map(|v| v.to_string())
            .unwrap_or_else(|| "-".to_string()),
    );
    append_plain_value(
        rows,
        "Pre-emphasis",
        track
            .preemphasis
            .map(|v| v.to_string())
            .unwrap_or_else(|| "-".to_string()),
    );
    rows.push(DetailRow::line(
        2,
        format!(
            "Integrity: {}",
            integrity_text(&track.test_and_copy.integrity)
        ),
        integrity_mark(&track.test_and_copy.integrity),
    ));
    rows.push(DetailRow::line(
        2,
        format!(
            "Integrity skip zeroes: {}",
            integrity_text(&track.test_and_copy.integrity_skipzero)
        ),
        integrity_mark(&track.test_and_copy.integrity_skipzero),
    ));
    append_plain_value(
        rows,
        "Test hash",
        empty_dash(&track.test_and_copy.test_hash),
    );
    append_plain_value(
        rows,
        "Copy hash",
        empty_dash(&track.test_and_copy.copy_hash),
    );
    append_plain_value(
        rows,
        "Test hash skip zeroes",
        empty_dash(&track.test_and_copy.test_skipzero_hash),
    );
    append_plain_value(
        rows,
        "Copy hash skip zeroes",
        empty_dash(&track.test_and_copy.copy_skipzero_hash),
    );
    append_track_error(rows, "Read errors", &track.errors.read);
    append_track_error(rows, "Skip errors", &track.errors.skip);
    append_track_error(rows, "Jitter errors", &track.errors.jitter_generic);
    append_track_error(rows, "Edge jitter errors", &track.errors.jitter_edge);
    append_track_error(rows, "Atom jitter errors", &track.errors.jitter_atom);
    append_track_error(rows, "Drift errors", &track.errors.drift);
    append_track_error(rows, "Dropped bytes", &track.errors.dropped);
    append_track_error(rows, "Duplicated bytes", &track.errors.duplicated);
    append_track_error(rows, "Damaged sectors", &track.errors.damaged_sectors);
    append_track_error(
        rows,
        "Inconsistent error sectors",
        &track.errors.inconsistent_err_sectors,
    );
    append_track_error(rows, "Missing samples", &track.errors.missing_samples);
}

fn append_track_error(rows: &mut Vec<DetailRow>, label: &str, data: &TrackErrorData) {
    if data.count == 0 && data.ranges.is_empty() {
        return;
    }
    rows.push(DetailRow::line(
        2,
        format!("{label}: {}", data.count),
        Mark::Bad,
    ));
    for range in &data.ranges {
        rows.push(DetailRow::line(
            3,
            format!(
                "start: {} | length: {}",
                time_text(&range.start),
                time_text(&range.length)
            ),
            Mark::Bad,
        ));
    }
}

fn append_plain_value(rows: &mut Vec<DetailRow>, label: &str, value: impl Into<String>) {
    rows.push(DetailRow::line(
        1,
        format!("{label}: {}", value.into()),
        Mark::Normal,
    ));
}

fn append_checked_value(
    rows: &mut Vec<DetailRow>,
    label: &str,
    value: impl Into<String>,
    evaluation: Option<&Evaluation>,
    fields: &[EvaluationUnitField],
) {
    let mark = fields_mark(evaluation, fields);
    let suffix = match mark {
        Mark::Good => " | 符合",
        Mark::Bad => " | 不符合",
        Mark::Warn => " | 未知",
        _ => "",
    };
    rows.push(DetailRow::line(
        1,
        format!("{label}: {}{suffix}", value.into()),
        mark,
    ));
}

fn fields_mark(evaluation: Option<&Evaluation>, fields: &[EvaluationUnitField]) -> Mark {
    let Some(evaluation) = evaluation else {
        return Mark::Warn;
    };
    let mut seen = false;
    let mut bad = false;
    let mut warn = false;
    for unit in &evaluation.evaluation_units {
        if fields.iter().any(|field| field == &unit.data.field) {
            seen = true;
            match unit_mark(&unit.data.class, &unit.unit_score) {
                Mark::Bad => bad = true,
                Mark::Warn => warn = true,
                _ => {}
            }
        }
    }
    if bad {
        Mark::Bad
    } else if warn {
        Mark::Warn
    } else if seen {
        Mark::Good
    } else {
        Mark::Good
    }
}

fn unit_mark(class: &EvaluationUnitClass, unit_score: &str) -> Mark {
    if matches!(
        class,
        EvaluationUnitClass::Critical | EvaluationUnitClass::Bad
    ) {
        return Mark::Bad;
    }
    let score = unit_score.trim().parse::<i32>().ok();
    match score {
        Some(v) if v < 0 => Mark::Bad,
        Some(0) => Mark::Warn,
        Some(100) => Mark::Good,
        Some(_)
            if matches!(
                class,
                EvaluationUnitClass::Good | EvaluationUnitClass::Perfect
            ) =>
        {
            Mark::Good
        }
        Some(_) => Mark::Warn,
        None => Mark::Warn,
    }
}

fn score_mark(score: &str) -> Mark {
    match score.trim().parse::<i32>() {
        Ok(100) => Mark::Good,
        Ok(v) if v < 0 => Mark::Bad,
        Ok(v) if v < 100 => Mark::Warn,
        Ok(_) => Mark::Good,
        Err(_) => Mark::Muted,
    }
}

fn track_mark(track: &TrackEntry) -> Mark {
    if track.aborted
        || track.test_and_copy.integrity == Integrity::Mismatch
        || track.test_and_copy.integrity_skipzero == Integrity::Mismatch
        || track_error_count(track) > 0
    {
        Mark::Bad
    } else if track.test_and_copy.integrity == Integrity::Match
        || track.test_and_copy.integrity_skipzero == Integrity::Match
    {
        Mark::Good
    } else {
        Mark::Warn
    }
}

fn track_error_count(track: &TrackEntry) -> u32 {
    track.errors.read.count
        + track.errors.skip.count
        + track.errors.jitter_generic.count
        + track.errors.jitter_edge.count
        + track.errors.jitter_atom.count
        + track.errors.drift.count
        + track.errors.dropped.count
        + track.errors.duplicated.count
        + track.errors.damaged_sectors.count
        + track.errors.inconsistent_err_sectors.count
        + track.errors.missing_samples.count
}

fn ar_status_mark(status: &AccurateRipStatus) -> Mark {
    match status {
        AccurateRipStatus::Match => Mark::Good,
        AccurateRipStatus::Mismatch => Mark::Bad,
        AccurateRipStatus::Offsetted | AccurateRipStatus::NotFound => Mark::Warn,
        AccurateRipStatus::Disabled => Mark::Muted,
    }
}

fn format_ar_offset(offset: &AccurateRipOffset) -> String {
    match offset {
        AccurateRipOffset::Same => "Same".to_string(),
        AccurateRipOffset::Different(value) => format!(
            "Different({})",
            value
                .map(|v| v.to_string())
                .unwrap_or_else(|| "-".to_string())
        ),
    }
}

fn track_list_items(entry: &LogEntry, parsed_index: usize) -> Vec<String> {
    let Some(log) = entry.response.parsed.parsed_logs.get(parsed_index) else {
        return vec!["Unknown Artist - Unknown Album".to_string()];
    };
    let mut items = vec![release_line(log)];
    items.extend(log.tracks.iter().map(track_summary));
    items
}

fn release_line(log: &ParsedLog) -> String {
    format!("{} - {}", log.release_info.artist, log.release_info.title)
}

fn track_summary(track: &TrackEntry) -> String {
    let filename = track
        .filenames
        .first()
        .map(|name| name.as_str())
        .unwrap_or("-");
    format!(
        "Track {} | {} | {} | {}",
        track.num,
        if track.is_range { "Range" } else { "Split" },
        integrity_text(&track.test_and_copy.integrity),
        filename
    )
}

fn format_scope(scope: &EvaluationUnitScope) -> String {
    match scope {
        EvaluationUnitScope::Release => "Release".to_string(),
        EvaluationUnitScope::Track(Some(track)) => format!("Track {track}"),
        EvaluationUnitScope::Track(None) => "Track".to_string(),
    }
}

fn empty_dash(value: &str) -> String {
    if value.is_empty() {
        "-".to_string()
    } else {
        value.to_string()
    }
}

#[cfg(target_os = "windows")]
fn open_file_location(path: &Path) -> std::io::Result<()> {
    let wide = path
        .as_os_str()
        .encode_wide()
        .chain(std::iter::once(0))
        .collect::<Vec<u16>>();
    unsafe {
        let full = ILCreateFromPathW(wide.as_ptr());
        if full.is_null() {
            return Err(std::io::Error::last_os_error());
        }

        let result = if path.is_file() {
            let folder = ILClone(full);
            if folder.is_null() {
                ILFree(full);
                return Err(std::io::Error::last_os_error());
            }
            if ILRemoveLastID(folder) == 0 {
                ILFree(folder);
                ILFree(full);
                return Err(std::io::Error::last_os_error());
            }
            let child = ILFindLastID(full);
            let items = [child as *const _];
            let hr = SHOpenFolderAndSelectItems(folder, 1, items.as_ptr(), 0);
            ILFree(folder);
            hr
        } else {
            SHOpenFolderAndSelectItems(full, 0, std::ptr::null(), 0)
        };

        ILFree(full);
        if result < 0 {
            Err(std::io::Error::from_raw_os_error(result))
        } else {
            Ok(())
        }
    }
}

fn format_optional<T: std::fmt::Display>(value: Option<T>) -> String {
    value
        .map(|v| v.to_string())
        .unwrap_or_else(|| "-".to_string())
}

#[cfg(target_os = "windows")]
#[derive(Clone)]
struct WindowRegistration {
    hwnd: isize,
    old_proc: WNDPROC,
    sender: Option<ComponentSender<MainModel>>,
}

#[cfg(target_os = "windows")]
#[derive(Clone, Copy)]
struct LabelColor {
    hwnd: isize,
    mark: Mark,
}

#[cfg(target_os = "windows")]
static WINDOW_REGISTRATIONS: OnceLock<Mutex<Vec<WindowRegistration>>> = OnceLock::new();
#[cfg(target_os = "windows")]
static LABEL_COLORS: OnceLock<Mutex<Vec<LabelColor>>> = OnceLock::new();

#[cfg(target_os = "windows")]
fn install_window_handler(window: &Window, sender: ComponentSender<MainModel>) {
    let hwnd = window.as_window().as_win32();
    unsafe {
        DragAcceptFiles(hwnd, 1);
        register_window_proc(hwnd as isize, Some(sender));
    }
}

#[cfg(target_os = "windows")]
fn install_color_parent_handler(scroll: &ScrollView) {
    register_window_proc(scroll.as_container().as_win32() as isize, None);
}

#[cfg(target_os = "windows")]
fn register_window_proc(hwnd: isize, sender: Option<ComponentSender<MainModel>>) {
    unsafe {
        let old = SetWindowLongPtrW(
            hwnd as HWND,
            GWLP_WNDPROC,
            window_proc as *const () as isize,
        );
        if old != 0 {
            let old_proc: WNDPROC = std::mem::transmute(old);
            WINDOW_REGISTRATIONS
                .get_or_init(|| Mutex::new(Vec::new()))
                .lock()
                .unwrap()
                .push(WindowRegistration {
                    hwnd,
                    old_proc,
                    sender,
                });
        }
    }
}

#[cfg(target_os = "windows")]
fn register_label_color(label: &Label, mark: Mark) {
    LABEL_COLORS
        .get_or_init(|| Mutex::new(Vec::new()))
        .lock()
        .unwrap()
        .push(LabelColor {
            hwnd: label.as_widget().as_win32() as isize,
            mark,
        });
}

#[cfg(target_os = "windows")]
fn clear_colored_labels() {
    if let Some(labels) = LABEL_COLORS.get() {
        if let Ok(mut labels) = labels.lock() {
            labels.clear();
        }
    }
}

#[cfg(windows)]
fn set_process_app_user_model_id(id: &str) -> Result<()> {
    use windows_sys::Win32::Foundation::S_OK;
    use windows_sys::Win32::UI::Shell::SetCurrentProcessExplicitAppUserModelID;

    let wide = id.encode_utf16().chain(std::iter::once(0)).collect::<Vec<_>>();
    let hr = unsafe { SetCurrentProcessExplicitAppUserModelID(wide.as_ptr()) };
    if hr == S_OK {
        Ok(())
    } else {
        Err(Error::empty())
    }
}

#[cfg(windows)]
fn set_window_app_user_model_properties(
    hwnd: HWND,
    app_user_model_id: &str,
    app_name: &str,
    icon_resource_id: Option<u16>,
) -> Result<()> {
    use std::ffi::c_void;
    use windows_sys::core::{GUID, HRESULT};
    use windows_sys::Win32::Foundation::{PROPERTYKEY, S_OK};
    use windows_sys::Win32::System::Com::StructuredStorage::PROPVARIANT;
    use windows_sys::Win32::System::Variant::VT_LPWSTR;
    use windows_sys::Win32::UI::Shell::PropertiesSystem::SHGetPropertyStoreForWindow;

    const IID_IPROPERTY_STORE: GUID = GUID::from_u128(0x886d8eeb_8cf2_4446_8d02_cdba1dbdcf99);
    const APP_USER_MODEL_PROPERTY_GUID: GUID =
        GUID::from_u128(0x9f4c2855_9f79_4b39_a8d0_e1d42de1d5f3);
    const PKEY_APPUSERMODEL_RELAUNCH_COMMAND: PROPERTYKEY = PROPERTYKEY {
        fmtid: APP_USER_MODEL_PROPERTY_GUID,
        pid: 2,
    };
    const PKEY_APPUSERMODEL_RELAUNCH_ICON_RESOURCE: PROPERTYKEY = PROPERTYKEY {
        fmtid: APP_USER_MODEL_PROPERTY_GUID,
        pid: 3,
    };
    const PKEY_APPUSERMODEL_RELAUNCH_DISPLAY_NAME_RESOURCE: PROPERTYKEY = PROPERTYKEY {
        fmtid: APP_USER_MODEL_PROPERTY_GUID,
        pid: 4,
    };
    const PKEY_APPUSERMODEL_ID: PROPERTYKEY = PROPERTYKEY {
        fmtid: APP_USER_MODEL_PROPERTY_GUID,
        pid: 5,
    };

    #[repr(C)]
    struct IPropertyStore {
        vtbl: *const IPropertyStoreVtbl,
    }

    #[repr(C)]
    struct IPropertyStoreVtbl {
        query_interface: unsafe extern "system" fn(
            *mut IPropertyStore,
            *const GUID,
            *mut *mut c_void,
        ) -> HRESULT,
        add_ref: unsafe extern "system" fn(*mut IPropertyStore) -> u32,
        release: unsafe extern "system" fn(*mut IPropertyStore) -> u32,
        get_count: unsafe extern "system" fn(*mut IPropertyStore, *mut u32) -> HRESULT,
        get_at: unsafe extern "system" fn(*mut IPropertyStore, u32, *mut PROPERTYKEY) -> HRESULT,
        get_value:
            unsafe extern "system" fn(*mut IPropertyStore, *const PROPERTYKEY, *mut PROPVARIANT)
                -> HRESULT,
        set_value:
            unsafe extern "system" fn(*mut IPropertyStore, *const PROPERTYKEY, *const PROPVARIANT)
                -> HRESULT,
        commit: unsafe extern "system" fn(*mut IPropertyStore) -> HRESULT,
    }

    fn set_string_property(
        store: *mut IPropertyStore,
        key: &PROPERTYKEY,
        value: &str,
    ) -> Result<()> {
        let mut wide = value.encode_utf16().chain(std::iter::once(0)).collect::<Vec<_>>();
        let mut prop = PROPVARIANT::default();
        prop.Anonymous.Anonymous.vt = VT_LPWSTR;
        prop.Anonymous.Anonymous.Anonymous.pwszVal = wide.as_mut_ptr();

        let hr = unsafe { ((*(*store).vtbl).set_value)(store, key, &prop) };
        if hr >= 0 {
            Ok(())
        } else {
            Err(Error::empty())
        }
    }

    let exe = std::env::current_exe().map_err(|_| Error::empty())?;
    let exe_path = exe.to_string_lossy();
    let relaunch_command = quote_windows_command_argument(&exe_path);
    let icon_resource = icon_resource_id.map(|id| format!("{exe_path},-{id}"));

    let mut store = std::ptr::null_mut::<c_void>();
    let hr = unsafe { SHGetPropertyStoreForWindow(hwnd, &IID_IPROPERTY_STORE, &mut store) };
    if hr != S_OK || store.is_null() {
        return Err(Error::empty());
    }

    let store = store.cast::<IPropertyStore>();
    let result = (|| {
        set_string_property(store, &PKEY_APPUSERMODEL_RELAUNCH_COMMAND, &relaunch_command)?;
        set_string_property(
            store,
            &PKEY_APPUSERMODEL_RELAUNCH_DISPLAY_NAME_RESOURCE,
            app_name,
        )?;
        if let Some(icon_resource) = icon_resource.as_deref() {
            set_string_property(
                store,
                &PKEY_APPUSERMODEL_RELAUNCH_ICON_RESOURCE,
                icon_resource,
            )?;
        }
        set_string_property(store, &PKEY_APPUSERMODEL_ID, app_user_model_id)?;

        let hr = unsafe { ((*(*store).vtbl).commit)(store) };
        if hr >= 0 {
            Ok(())
        } else {
            Err(Error::empty())
        }
    })();

    unsafe {
        ((*(*store).vtbl).release)(store);
    }

    result
}

#[cfg(windows)]
fn quote_windows_command_argument(arg: &str) -> String {
    let mut result = String::from("\"");
    let mut backslashes = 0;

    for ch in arg.chars() {
        if ch == '\\' {
            backslashes += 1;
            continue;
        }

        if ch == '"' {
            result.extend(std::iter::repeat('\\').take(backslashes * 2 + 1));
            result.push('"');
            backslashes = 0;
            continue;
        }

        result.extend(std::iter::repeat('\\').take(backslashes));
        backslashes = 0;
        result.push(ch);
    }

    result.extend(std::iter::repeat('\\').take(backslashes * 2));
    result.push('"');
    result
}

#[cfg(windows)]
fn set_window_icons(hwnd: windows_sys::Win32::Foundation::HWND, icon_id: u16) -> Result<()> {
    use std::ptr::null;
    use windows_sys::Win32::System::LibraryLoader::GetModuleHandleW;
    use windows_sys::Win32::UI::WindowsAndMessaging::{
        GetSystemMetrics, LoadImageW, SendMessageW, ICON_BIG, ICON_SMALL, IMAGE_ICON,
        LR_DEFAULTCOLOR, LR_SHARED, SM_CXICON, SM_CXSMICON, SM_CYICON, SM_CYSMICON, WM_SETICON,
    };

    unsafe {
        let hinst = GetModuleHandleW(null());

        let big = LoadImageW(
            hinst,
            icon_id as _,
            IMAGE_ICON,
            GetSystemMetrics(SM_CXICON),
            GetSystemMetrics(SM_CYICON),
            LR_DEFAULTCOLOR | LR_SHARED,
        );

        let small = LoadImageW(
            hinst,
            icon_id as _,
            IMAGE_ICON,
            GetSystemMetrics(SM_CXSMICON),
            GetSystemMetrics(SM_CYSMICON),
            LR_DEFAULTCOLOR | LR_SHARED,
        );

        if big.is_null() || small.is_null() {
            panic!("cannot set icon");
        }

        SendMessageW(hwnd, WM_SETICON, ICON_BIG as _, big as _);
        SendMessageW(hwnd, WM_SETICON, ICON_SMALL as _, small as _);
    }

    Ok(())
}

#[cfg(windows)]
pub fn raise_once(hwnd: windows_sys::Win32::Foundation::HWND) -> Result<()> {
    use windows_sys::Win32::UI::WindowsAndMessaging::{
        SetWindowPos, HWND_NOTOPMOST, HWND_TOPMOST, SWP_NOMOVE, SWP_NOSIZE, SWP_SHOWWINDOW,
    };

    let flags = SWP_NOMOVE | SWP_NOSIZE | SWP_SHOWWINDOW;

    unsafe {
        if SetWindowPos(hwnd, HWND_TOPMOST, 0, 0, 0, 0, flags) == 0 {
            panic!("cannot to top")
        }

        if SetWindowPos(hwnd, HWND_NOTOPMOST, 0, 0, 0, 0, flags) == 0 {
            panic!("cannot reset")
        }
    }

    Ok(())
}

#[cfg(target_os = "windows")]
unsafe extern "system" fn window_proc(
    hwnd: HWND,
    msg: u32,
    wparam: WPARAM,
    lparam: LPARAM,
) -> LRESULT {
    if msg == WM_DROPFILES {
        let paths = dropped_paths(wparam as HDROP);
        if !paths.is_empty() {
            if let Some(registration) = find_window_registration(hwnd) {
                if let Some(sender) = registration.sender {
                    sender.post(MainMessage::LoadPaths(paths));
                }
            }
        }
        DragFinish(wparam as HDROP);
        return 0;
    }

    if msg == WM_CTLCOLORSTATIC {
        if let Some(mark) = find_label_mark(lparam as HWND) {
            // Let winio's original proc apply the theme-correct background
            // (dark brush + SetBkColor in dark mode, white otherwise), then
            // override only the text color with our semantic mark color.
            let brush = if let Some(registration) = find_window_registration(hwnd) {
                CallWindowProcW(registration.old_proc, hwnd, msg, wparam, lparam)
            } else {
                DefWindowProcW(hwnd, msg, wparam, lparam)
            };
            if let Some(color) = mark.color(is_dark_paint(wparam as HDC)) {
                SetTextColor(wparam as HDC, color);
            }
            return brush;
        }
    }

    call_original_window_proc(hwnd, msg, wparam, lparam)
}

#[cfg(target_os = "windows")]
unsafe fn call_original_window_proc(
    hwnd: HWND,
    msg: u32,
    wparam: WPARAM,
    lparam: LPARAM,
) -> LRESULT {
    if let Some(registration) = find_window_registration(hwnd) {
        CallWindowProcW(registration.old_proc, hwnd, msg, wparam, lparam)
    } else {
        DefWindowProcW(hwnd, msg, wparam, lparam)
    }
}

#[cfg(target_os = "windows")]
fn find_window_registration(hwnd: HWND) -> Option<WindowRegistration> {
    WINDOW_REGISTRATIONS.get().and_then(|registrations| {
        registrations.lock().ok().and_then(|items| {
            items
                .iter()
                .find(|item| item.hwnd == hwnd as isize)
                .cloned()
        })
    })
}

#[cfg(target_os = "windows")]
fn find_label_mark(hwnd: HWND) -> Option<Mark> {
    LABEL_COLORS.get().and_then(|labels| {
        labels.lock().ok().and_then(|items| {
            items
                .iter()
                .find(|item| item.hwnd == hwnd as isize)
                .map(|item| item.mark)
        })
    })
}

#[cfg(target_os = "windows")]
fn is_dark_paint(hdc: HDC) -> bool {
    let background = unsafe { GetBkColor(hdc) };
    let text = unsafe { GetTextColor(hdc) };
    color_luminance(background) < 96 || color_luminance(text) > 192
}

#[cfg(target_os = "windows")]
fn color_luminance(color: u32) -> u32 {
    let r = color & 0xff;
    let g = (color >> 8) & 0xff;
    let b = (color >> 16) & 0xff;
    (r * 299 + g * 587 + b * 114) / 1000
}

#[cfg(target_os = "windows")]
unsafe fn dropped_paths(drop: HDROP) -> Vec<PathBuf> {
    let count = DragQueryFileW(drop, u32::MAX, std::ptr::null_mut(), 0);
    let mut paths = Vec::new();
    for index in 0..count {
        let len = DragQueryFileW(drop, index, std::ptr::null_mut(), 0);
        if len == 0 {
            continue;
        }
        let mut buffer = vec![0u16; len as usize + 1];
        let copied = DragQueryFileW(drop, index, buffer.as_mut_ptr(), buffer.len() as u32);
        if copied > 0 {
            buffer.truncate(copied as usize);
            paths.push(PathBuf::from(String::from_utf16_lossy(&buffer)));
        }
    }
    paths
}
