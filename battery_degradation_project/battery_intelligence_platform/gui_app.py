"""
Battery Intelligence Platform - desktop GUI.

Lets you pick a battery dataset, runs the existing analysis pipeline
(core/a_capability_engine.py) in the background, and shows the summary,
the charts, and the run log. Uses only the Python standard library
(tkinter) plus the packages in requirements.txt.

Run from this folder:
    python gui_app.py
"""

import csv
import os
import subprocess
import sys
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

PLATFORM_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = PLATFORM_DIR / "outputs"
PIPELINE_SCRIPT = PLATFORM_DIR / "core" / "a_capability_engine.py"
PROJECT_DATA_DIR = PLATFORM_DIR.parent  # where the sample battery CSVs live
SUMMARY_CSV = OUTPUT_DIR / "reports" / "battery_summary.csv"
REPORT_TXT = OUTPUT_DIR / "reports" / "battery_report.txt"

CHARTS = {
    "SOH vs cycle": OUTPUT_DIR / "graphs" / "soh_vs_cycle.png",
    "Capacity vs cycle": OUTPUT_DIR / "graphs" / "capacity_vs_cycle.png",
    "ICA (dQ/dV)": OUTPUT_DIR / "ica_analysis" / "ica_analysis.png",
    "DVA (dV/dQ)": OUTPUT_DIR / "dva_analysis" / "dva_analysis.png",
    "SOC estimate": OUTPUT_DIR / "soc_analysis" / "soc_analysis.png",
}

MAX_IMAGE_WIDTH = 900
MAX_IMAGE_HEIGHT = 600


def _format_value(text):
    """Round long decimals to 2 places so the summary is easy to read."""
    try:
        number = float(text)
    except (TypeError, ValueError):
        return text
    if number.is_integer():
        return str(int(number))
    return f"{number:.2f}"


class BatteryApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Battery Intelligence Platform")
        self.geometry("1000x720")
        self.minsize(800, 600)

        self.dataset_var = tk.StringVar()
        self.status_var = tk.StringVar(value="Choose a dataset and press Run.")
        self.chart_var = tk.StringVar(value=list(CHARTS)[0])
        self.photo = None  # keep a reference so Tk does not discard the image
        self.running = False

        self._build_widgets()
        self._show_existing_results()

    # ------------------------------------------------------------ layout
    def _build_widgets(self):
        top = ttk.Frame(self, padding=10)
        top.pack(fill="x")

        ttk.Label(top, text="Dataset (CSV or Excel):").pack(side="left")
        entry = ttk.Entry(top, textvariable=self.dataset_var, width=70)
        entry.pack(side="left", padx=6, fill="x", expand=True)
        ttk.Button(top, text="Browse...", command=self.browse).pack(side="left")
        self.run_button = ttk.Button(top, text="Run analysis", command=self.start_run)
        self.run_button.pack(side="left", padx=6)

        ttk.Label(self, textvariable=self.status_var, padding=(10, 0)).pack(fill="x")

        self.notebook = ttk.Notebook(self)
        self.notebook.pack(fill="both", expand=True, padx=10, pady=10)

        # Summary tab
        summary_tab = ttk.Frame(self.notebook, padding=10)
        self.notebook.add(summary_tab, text="Summary")
        self.summary_table = ttk.Treeview(
            summary_tab, columns=("metric", "value"), show="headings", height=10
        )
        self.summary_table.heading("metric", text="Metric")
        self.summary_table.heading("value", text="Value")
        self.summary_table.column("metric", width=260, anchor="w")
        self.summary_table.column("value", width=500, anchor="w")
        self.summary_table.pack(fill="x")
        ttk.Label(summary_tab, text="Full report:").pack(anchor="w", pady=(10, 2))
        self.report_text = tk.Text(summary_tab, height=14, wrap="word")
        self.report_text.pack(fill="both", expand=True)

        # Charts tab
        chart_tab = ttk.Frame(self.notebook, padding=10)
        self.notebook.add(chart_tab, text="Charts")
        chart_bar = ttk.Frame(chart_tab)
        chart_bar.pack(fill="x")
        ttk.Label(chart_bar, text="Chart:").pack(side="left")
        chart_box = ttk.Combobox(
            chart_bar,
            textvariable=self.chart_var,
            values=list(CHARTS),
            state="readonly",
            width=30,
        )
        chart_box.pack(side="left", padx=6)
        chart_box.bind("<<ComboboxSelected>>", lambda _e: self.show_chart())
        ttk.Button(chart_bar, text="Open chart file", command=self.open_chart_file).pack(
            side="left", padx=6
        )
        self.chart_label = ttk.Label(chart_tab, anchor="center")
        self.chart_label.pack(fill="both", expand=True, pady=10)

        # Log tab
        log_tab = ttk.Frame(self.notebook, padding=10)
        self.notebook.add(log_tab, text="Run log")
        self.log_text = tk.Text(log_tab, wrap="word", font=("Consolas", 9))
        self.log_text.pack(fill="both", expand=True)

        # Bottom bar
        bottom = ttk.Frame(self, padding=(10, 0, 10, 10))
        bottom.pack(fill="x")
        ttk.Button(bottom, text="Open outputs folder", command=self.open_outputs).pack(
            side="left"
        )
        ttk.Button(bottom, text="Refresh results", command=self._show_existing_results).pack(
            side="left", padx=6
        )

    # ----------------------------------------------------------- actions
    def browse(self):
        path = filedialog.askopenfilename(
            title="Choose a battery dataset",
            filetypes=[
                ("Battery data", "*.csv *.xlsx *.xls"),
                ("CSV files", "*.csv"),
                ("Excel files", "*.xlsx *.xls"),
                ("All files", "*.*"),
            ],
        )
        if path:
            self.dataset_var.set(path)

    def start_run(self):
        if self.running:
            return
        typed = self.dataset_var.get().strip().strip('"')
        if not typed:
            messagebox.showwarning("No dataset", "Choose a dataset file first.")
            return
        # Accept a full path, or a bare file name from the project's data folder.
        candidates = [Path(typed), PROJECT_DATA_DIR / typed]
        dataset_path = next((c for c in candidates if c.is_file()), None)
        if dataset_path is None:
            messagebox.showerror(
                "File not found",
                f"Cannot find:\n{typed}\n\nUse Browse... to pick the file.",
            )
            return
        dataset = str(dataset_path.resolve())
        self.dataset_var.set(dataset)

        self.running = True
        self.run_button.config(state="disabled")
        self.status_var.set("Running analysis... this can take a few minutes.")
        self._set_log("")
        self.notebook.select(2)  # show the Run log tab while it runs

        thread = threading.Thread(target=self._run_pipeline, args=(dataset,), daemon=True)
        thread.start()

    def _run_pipeline(self, dataset):
        env = dict(os.environ, PYTHONIOENCODING="utf-8")
        try:
            result = subprocess.run(
                [sys.executable, str(PIPELINE_SCRIPT)],
                input=dataset + "\n",
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                cwd=str(PLATFORM_DIR),
                env=env,
            )
            output = (result.stdout or "") + (result.stderr or "")
            # The pipeline can exit with code 0 without analysing anything
            # (for example, when it gets no valid path), so require the
            # report to have been written.
            ok = result.returncode == 0 and "Report saved:" in output
        except Exception as error:  # noqa: BLE001 - show any failure to the user
            output = f"Could not start the analysis: {error}"
            ok = False
        self.after(0, self._run_finished, ok, output)

    def _run_finished(self, ok, output):
        self.running = False
        self.run_button.config(state="normal")
        self._set_log(output)
        self._show_existing_results()
        if ok:
            self.status_var.set("Analysis finished. See the Summary and Charts tabs.")
            self.notebook.select(0)
        else:
            self.status_var.set("Analysis failed. See the Run log tab.")
            messagebox.showerror("Analysis failed", "The run did not finish. Check the Run log tab.")

    # ----------------------------------------------------------- display
    def _set_log(self, text):
        self.log_text.config(state="normal")
        self.log_text.delete("1.0", "end")
        self.log_text.insert("1.0", text)
        self.log_text.config(state="disabled")

    def _show_existing_results(self):
        self._load_summary()
        self._load_report()
        self.show_chart()

    def _load_summary(self):
        self.summary_table.delete(*self.summary_table.get_children())
        if not SUMMARY_CSV.exists():
            self.summary_table.insert("", "end", values=("No results yet", "Run an analysis"))
            return
        with SUMMARY_CSV.open(encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                self.summary_table.insert(
                    "", "end", values=(row.get("Metric", ""), _format_value(row.get("Value", "")))
                )

    def _load_report(self):
        self.report_text.config(state="normal")
        self.report_text.delete("1.0", "end")
        if REPORT_TXT.exists():
            self.report_text.insert("1.0", REPORT_TXT.read_text(encoding="utf-8", errors="replace"))
        self.report_text.config(state="disabled")

    def show_chart(self):
        path = CHARTS[self.chart_var.get()]
        if not path.exists():
            self.chart_label.config(image="", text=f"No chart yet:\n{path.name}")
            self.photo = None
            return
        image = tk.PhotoImage(file=str(path))
        # Shrink large images by whole-number steps so they fit the window.
        factor = max(
            1,
            -(-image.width() // MAX_IMAGE_WIDTH),
            -(-image.height() // MAX_IMAGE_HEIGHT),
        )
        if factor > 1:
            image = image.subsample(factor, factor)
        self.photo = image
        self.chart_label.config(image=self.photo, text="")

    def open_chart_file(self):
        self._open_path(CHARTS[self.chart_var.get()])

    def open_outputs(self):
        self._open_path(OUTPUT_DIR)

    def _open_path(self, path):
        if not path.exists():
            messagebox.showinfo("Not found", f"Nothing to open yet:\n{path}")
            return
        if hasattr(os, "startfile"):
            os.startfile(str(path))  # Windows
        else:
            subprocess.Popen(["xdg-open", str(path)])


if __name__ == "__main__":
    BatteryApp().mainloop()
