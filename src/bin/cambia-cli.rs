use std::fs::{self, File};
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::{
    atomic::{AtomicUsize, Ordering},
    Arc, Mutex,
};
use std::thread;
use std::time::Duration;

use clap::Parser;

use cambia_core::evaluate::{EvaluationUnitScope, EvaluatorType};
use cambia_core::handler::parse_log_bytes;
use cambia_core::response::CambiaResponse;
use ratatui::backend::CrosstermBackend;
use ratatui::crossterm::event::{self, Event, KeyCode, KeyEventKind, KeyModifiers};
use ratatui::crossterm::execute;
use ratatui::crossterm::terminal::{
    disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen,
};
use ratatui::layout::{Constraint, Direction, Layout};
use ratatui::prelude::{Backend, Rect};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::Line;
use ratatui::widgets::{
    Block, Borders, Cell, List, ListItem, ListState, Paragraph, Row, Table, Wrap,
};
use ratatui::{Frame, Terminal};
use rayon::prelude::*;
use walkdir::WalkDir;

/// Cambia CLI - parse CD ripping logs locally
#[derive(Parser, Debug)]
#[command(name = "cambia-cli", author, version, about = "CD ripper log checker", long_about = None)]
struct Cli {
    /// Path to the rip log file or directory
    #[arg(value_name = "PATH", value_hint = clap::ValueHint::AnyPath)]
    path: PathBuf,

    /// Save parsed log bytes to the specified directory
    #[arg(long, value_name = "DIR", value_hint = clap::ValueHint::DirPath)]
    save_logs: Option<PathBuf>,

    /// Set tracing level (trace|debug|info|warn|error)
    #[arg(long, default_value = "info")]
    tracing: String,

    /// 显示 OPS 扣分为 100 的条目
    #[arg(long = "show-100")]
    show_100: bool,
}

fn main() {
    let cli = Cli::parse();

    init_logging(&cli.tracing);

    if let Err(err) = run(cli) {
        eprintln!("{err}");
        std::process::exit(1);
    }
}

fn run(cli: Cli) -> Result<(), String> {
    let metadata = fs::metadata(&cli.path)
        .map_err(|err| format!("无法访问路径 {}: {err}", cli.path.display()))?;

    let mut log_paths: Vec<PathBuf> = Vec::new();

    if metadata.is_dir() {
        for entry in WalkDir::new(&cli.path).into_iter().filter_map(Result::ok) {
            let path = entry.path();
            if entry.file_type().is_file() {
                if let Some(ext) = path.extension().and_then(|s| s.to_str()) {
                    if ext.eq_ignore_ascii_case("log") {
                        log_paths.push(path.to_path_buf());
                    }
                }
            }
        }

        if log_paths.is_empty() {
            return Err(format!(
                "目录 {} 中未找到任何 .log 文件",
                cli.path.display()
            ));
        }
    } else if metadata.is_file() {
        log_paths.push(cli.path.clone());
    } else {
        return Err(format!("路径 {} 既不是文件也不是目录", cli.path.display()));
    }

    let total = log_paths.len();
    let counter = Arc::new(AtomicUsize::new(0));
    let shared_logs: Arc<Mutex<Vec<LogEntry>>> = Arc::new(Mutex::new(Vec::new()));

    let save_logs = cli.save_logs.clone();
    let show_100 = cli.show_100;

    let counter_worker = Arc::clone(&counter);
    let logs_worker = Arc::clone(&shared_logs);

    // 多线程分析日志
    let handle = thread::spawn(move || {
        log_paths.into_par_iter().for_each(|path| {
            match analyze_single_log(path, &save_logs) {
                Ok(entry) => {
                    if let Ok(mut logs) = logs_worker.lock() {
                        logs.push(entry);
                    }
                }
                Err(err) => {
                    eprintln!("{err}");
                }
            }

            counter_worker.fetch_add(1, Ordering::Relaxed);
        });
    });

    while counter.load(Ordering::Relaxed) < total {
        let done = counter.load(Ordering::Relaxed);
        let percent = if total > 0 { done * 100 / total } else { 100 };
        eprint!("\r分析中: {done}/{total} ({percent}%)");
        thread::sleep(Duration::from_millis(100));
    }

    handle
        .join()
        .map_err(|_| "分析线程发生 panic".to_string())?;

    eprintln!();

    let mut logs = shared_logs
        .lock()
        .map_err(|_| "无法获取分析结果".to_string())
        .map(|mut guard| std::mem::take(&mut *guard))?;

    // 如果未开启 --show-100，则在文件列表中隐藏 OPS 总分为 100 的日志
    if !show_100 {
        logs.retain(|entry| !is_ops_full_score(&entry.response));
    }

    render_ui(logs, show_100)?;

    Ok(())
}

fn init_logging(tracing: &str) {
    let tracing_level = match tracing.to_ascii_lowercase().as_str() {
        "trace" => tracing::Level::TRACE,
        "debug" => tracing::Level::DEBUG,
        "warn" => tracing::Level::WARN,
        "error" => tracing::Level::ERROR,
        _ => tracing::Level::INFO,
    };

    tracing_subscriber::fmt()
        .with_max_level(tracing_level)
        .init();
}

struct LogEntry {
    path: PathBuf,
    response: CambiaResponse,
}

struct App {
    logs: Vec<LogEntry>,
    list_state: ListState,
    show_100: bool,
}

impl App {
    fn new(logs: Vec<LogEntry>, show_100: bool) -> Self {
        let mut list_state = ListState::default();
        if !logs.is_empty() {
            list_state.select(Some(0));
        }
        Self {
            logs,
            list_state,
            show_100,
        }
    }

    fn select_next(&mut self) {
        if self.logs.is_empty() {
            return;
        }
        let i = match self.list_state.selected() {
            Some(i) if i + 1 < self.logs.len() => i + 1,
            _ => self.logs.len() - 1,
        };
        self.list_state.select(Some(i));
    }

    fn select_previous(&mut self) {
        if self.logs.is_empty() {
            return;
        }
        let i = match self.list_state.selected() {
            Some(i) if i > 0 => i - 1,
            _ => 0,
        };
        self.list_state.select(Some(i));
    }
}

fn render_ui(logs: Vec<LogEntry>, show_100: bool) -> Result<(), String> {
    let mut stdout = io::stdout();
    enable_raw_mode().map_err(|err| format!("无法进入原始模式: {err}"))?;
    execute!(stdout, EnterAlternateScreen).map_err(|err| format!("无法切换到备用屏幕: {err}"))?;
    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend).map_err(|err| format!("无法初始化终端: {err}"))?;
    terminal
        .hide_cursor()
        .map_err(|err| format!("无法隐藏光标: {err}"))?;

    let mut app = App::new(logs, show_100);

    let ui_result = ui_loop(&mut terminal, &mut app);

    terminal.show_cursor().ok();
    execute!(terminal.backend_mut(), LeaveAlternateScreen).ok();
    disable_raw_mode().ok();

    ui_result
}

fn ui_loop<B: Backend>(terminal: &mut Terminal<B>, app: &mut App) -> Result<(), String> {
    loop {
        terminal
            .draw(|frame| draw_frame(frame, app))
            .map_err(|err| format!("渲染界面失败: {err}"))?;

        match event::read().map_err(|err| format!("读取输入失败: {err}"))? {
            Event::Key(key) if key.kind == KeyEventKind::Press => match key.code {
                KeyCode::Char('q') | KeyCode::Esc | KeyCode::Enter => break,
                KeyCode::Char('c') if key.modifiers.contains(KeyModifiers::CONTROL) => {
                    break;
                }
                KeyCode::Down | KeyCode::Char('j') => app.select_next(),
                KeyCode::Up | KeyCode::Char('k') => app.select_previous(),
                KeyCode::Char('o') => {
                    if let Some(idx) = app.list_state.selected() {
                        if let Some(entry) = app.logs.get(idx) {
                            if let Err(err) = open_in_file_manager(&entry.path) {
                                tracing::error!("打开目录失败 {}: {}", entry.path.display(), err);
                            }
                        }
                    }
                }
                _ => {}
            },
            Event::Resize(_, _) => {}
            _ => {}
        }
    }

    Ok(())
}

fn draw_frame(frame: &mut Frame<'_>, app: &mut App) {
    let layout = Layout::default()
        .direction(Direction::Vertical)
        .margin(1)
        .constraints([
            Constraint::Length(5),
            Constraint::Min(0),
            Constraint::Length(5),
            Constraint::Length(3),
        ])
        .split(frame.area());

    if app.logs.is_empty() {
        let empty = Paragraph::new("没有可显示的日志")
            .style(Style::default().fg(Color::Gray))
            .block(Block::default().title("概览").borders(Borders::ALL));
        frame.render_widget(empty, layout[0]);

        let help = Paragraph::new("按 q / Esc / Enter 退出")
            .style(Style::default().fg(Color::Gray))
            .block(Block::default().title("帮助").borders(Borders::ALL));
        frame.render_widget(help, layout[3]);
        return;
    }

    let selected = app
        .list_state
        .selected()
        .unwrap_or(0)
        .min(app.logs.len() - 1);

    let body = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Percentage(30), Constraint::Percentage(70)])
        .split(layout[1]);

    render_file_list(frame, body[0], app);

    let entry = &app.logs[selected];

    render_summary(frame, layout[0], &entry.path, &entry.response);

    render_details(frame, body[1], &entry.response, app.show_100);
    render_score_table(frame, layout[2], &entry.response);

    let help = Paragraph::new("按 ↑/↓ 或 j/k 切换文件，按 o 打开所在目录，按 q / Esc / Enter 退出")
        .style(Style::default().fg(Color::Gray))
        .block(Block::default().title("帮助").borders(Borders::ALL));
    frame.render_widget(help, layout[3]);
}

fn render_file_list(frame: &mut Frame<'_>, area: Rect, app: &mut App) {
    let items: Vec<ListItem> = app
        .logs
        .iter()
        .map(|entry| {
            let name = entry
                .path
                .file_name()
                .and_then(|s| s.to_str())
                .unwrap_or("");
            let display = if name.is_empty() {
                entry.path.display().to_string()
            } else {
                name.to_string()
            };
            ListItem::new(display)
        })
        .collect();

    let list = List::new(items)
        .block(Block::default().title("日志文件").borders(Borders::ALL))
        .highlight_style(
            Style::default()
                .fg(Color::Cyan)
                .add_modifier(Modifier::BOLD),
        )
        .highlight_symbol("▶ ");

    frame.render_stateful_widget(list, area, &mut app.list_state);
}

fn render_summary(frame: &mut Frame<'_>, area: Rect, path: &Path, parsed: &CambiaResponse) {
    let summary_lines = vec![
        Line::from(format!("文件: {}", path.display())),
        Line::from(format!("Log ID: {}", hex::encode(&parsed.id))),
        Line::from(format!("评估器数量: {}", parsed.evaluation_combined.len())),
    ];

    let summary = Paragraph::new(summary_lines)
        .block(Block::default().title("概览").borders(Borders::ALL))
        .wrap(Wrap { trim: true });
    frame.render_widget(summary, area);
}

fn render_score_table(frame: &mut Frame<'_>, area: Rect, parsed: &CambiaResponse) {
    let header = Row::new(vec!["Evaluator", "Score", "Logs"])
        .style(Style::default().add_modifier(Modifier::BOLD));

    let mut rows: Vec<Row> = parsed
        .evaluation_combined
        .iter()
        .map(|evaluation| {
            Row::new(vec![
                Cell::from(format!("{:?}", evaluation.evaluator)),
                Cell::from(evaluation.combined_score.clone()),
                Cell::from(evaluation.evaluations.len().to_string()),
            ])
        })
        .collect();

    if rows.is_empty() {
        rows.push(Row::new(vec![
            Cell::from("N/A"),
            Cell::from("-"),
            Cell::from("-"),
        ]));
    }

    let table = Table::new(
        rows,
        [
            Constraint::Percentage(40),
            Constraint::Percentage(30),
            Constraint::Percentage(30),
        ],
    )
    .header(header)
    .block(Block::default().title("评估汇总").borders(Borders::ALL))
    .column_spacing(1);

    frame.render_widget(table, area);
}

fn render_details(frame: &mut Frame<'_>, area: Rect, parsed: &CambiaResponse, show_100: bool) {
    let detail_lines = build_detail_lines(parsed, show_100);
    let paragraph = Paragraph::new(detail_lines)
        .block(Block::default().title("详细扣分").borders(Borders::ALL))
        .wrap(Wrap { trim: false });

    frame.render_widget(paragraph, area);
}

fn build_detail_lines(parsed: &CambiaResponse, show_100: bool) -> Vec<Line<'static>> {
    let mut lines: Vec<Line<'static>> = Vec::new();

    for evaluation in &parsed.evaluation_combined {
        lines.push(Line::styled(
            format!(
                "{:?} (总分: {})",
                evaluation.evaluator, evaluation.combined_score
            ),
            Style::default()
                .fg(Color::Yellow)
                .add_modifier(Modifier::BOLD),
        ));

        for (idx, eval) in evaluation.evaluations.iter().enumerate() {
            lines.push(Line::styled(
                format!("  Log #{:<3} Score: {}", idx + 1, eval.score),
                Style::default().fg(Color::Cyan),
            ));

            for unit in &eval.evaluation_units {
                if !show_100
                    && matches!(evaluation.evaluator, EvaluatorType::OPS)
                    && unit.unit_score == "100"
                {
                    continue;
                }
                let scope = format_scope(&unit.data.scope);
                lines.push(Line::from(format!(
                    "    - [{}][{:?} {:?}] {} ({} 分)",
                    scope, unit.data.field, unit.data.class, unit.data.message, unit.unit_score
                )));
            }
        }

        lines.push(Line::default());
    }

    if lines.is_empty() {
        lines.push(Line::from("没有可显示的评估结果"));
    }

    lines
}

fn format_scope(scope: &EvaluationUnitScope) -> String {
    match scope {
        EvaluationUnitScope::Release => "Release".to_string(),
        EvaluationUnitScope::Track(Some(track)) => format!("Track {track}"),
        EvaluationUnitScope::Track(None) => "Track".to_string(),
    }
}

fn analyze_single_log(path: PathBuf, save_logs: &Option<PathBuf>) -> Result<LogEntry, String> {
    let bytes = fs::read(&path).map_err(|err| format!("无法读取文件 {}: {err}", path.display()))?;
    let parsed = parse_log_bytes(Vec::new(), &bytes)
        .map_err(|err| format!("解析日志失败 {}: {err}", path.display()))?;

    if let Some(ref save_dir) = save_logs {
        if let Err(err) = save_rip_log(save_dir, &parsed.id, &bytes) {
            eprintln!("保存日志失败 ({}): {err}", path.display());
        }
    }

    Ok(LogEntry {
        path,
        response: parsed,
    })
}

#[cfg(target_os = "windows")]
fn open_in_file_manager(path: &Path) -> Result<(), String> {
    let target = if path.is_file() {
        path.parent().unwrap_or(path)
    } else {
        path
    };

    Command::new("explorer")
        .arg(target)
        .spawn()
        .map(|_| ())
        .map_err(|err| format!("无法打开目录 {}: {err}", target.display()))
}

#[cfg(target_os = "macos")]
fn open_in_file_manager(path: &Path) -> Result<(), String> {
    let target = if path.is_file() {
        path.parent().unwrap_or(path)
    } else {
        path
    };

    Command::new("open")
        .arg(target)
        .spawn()
        .map(|_| ())
        .map_err(|err| format!("无法打开目录 {}: {err}", target.display()))
}

#[cfg(all(unix, not(target_os = "macos")))]
fn open_in_file_manager(path: &Path) -> Result<(), String> {
    let target = if path.is_file() {
        path.parent().unwrap_or(path)
    } else {
        path
    };

    Command::new("xdg-open")
        .arg(target)
        .spawn()
        .map(|_| ())
        .map_err(|err| format!("无法打开目录 {}: {err}", target.display()))
}

#[cfg(not(any(target_os = "windows", target_os = "macos", unix)))]
fn open_in_file_manager(_path: &Path) -> Result<(), String> {
    Err("当前平台不支持从 CLI 打开目录".to_string())
}

fn is_ops_full_score(resp: &CambiaResponse) -> bool {
    resp.evaluation_combined
        .iter()
        .find(|e| matches!(e.evaluator, EvaluatorType::OPS))
        .map(|e| e.combined_score.trim() == "100")
        .unwrap_or(false)
}

fn save_rip_log(root_path: &Path, id: &[u8], log_raw: &[u8]) -> Result<(), String> {
    fs::create_dir_all(root_path)
        .map_err(|err| format!("创建目录失败 {}: {err}", root_path.display()))?;

    let file_path = root_path.join(hex::encode(id)).with_extension("log");

    if file_path.exists() {
        return Ok(());
    }

    File::create(&file_path)
        .and_then(|mut file| file.write_all(log_raw))
        .map_err(|err| format!("写入日志失败 {}: {err}", file_path.display()))
}
