"""
Battery Intelligence Platform - desktop GUI.

Lets you pick a battery dataset, runs the existing analysis pipeline
(core/a_capability_engine.py) in the background, and shows the summary,
the charts, and the run log. Each dataset's results are kept in
outputs/<dataset name>/, so the summary and charts always belong to the
dataset shown in the Dataset box. Uses only the Python standard library
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

CHART_NAMES = [
    "SOH vs cycle",
    "Capacity vs cycle",
    "ICA (dQ/dV)",
    "DVA (dV/dQ)",
    "SOC estimate",
]

MAX_IMAGE_WIDTH = 900
MAX_IMAGE_HEIGHT = 600


def run_folder_for(dataset_path):
    """Folder that holds the results for one dataset."""
    return OUTPUT_DIR / Path(dataset_path).stem


def charts_for(run_dir):
    """Chart file for each chart name, inside one dataset's folder."""
    return {
        "SOH vs cycle": run_dir / "graphs" / "soh_vs_cycle.png",
        "Capacity vs cycle": run_dir / "graphs" / "capacity_vs_cycle.png",
        "ICA (dQ/dV)": run_dir / "ica_analysis" / "ica_analysis.png",
        "DVA (dV/dQ)": run_dir / "dva_analysis" / "dva_analysis.png",
        "SOC estimate": run_dir / "soc_analysis" / "soc_analysis.png",
    }


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
        self.chart_var = tk.StringVar(value=CHART_NAMES[0])
        self.photo = None  # keep a reference so Tk does not discard the image
        self.running = False
        self.run_dir = None  # results folder of the dataset currently shown

        self._build_widgets()
        self._select_dataset_from_box()
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
            values=CHART_NAMES,
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
        ttk.Button(bottom, text="Open results folder", command=self.open_outputs).pack(
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
            self._select_dataset_from_box()
            self._show_existing_results()

    def _select_dataset_from_box(self):
        """Point the results view at the dataset named in the Dataset box."""
        typed = self.dataset_var.get().strip().strip('"')
        if not typed:
            self.run_dir = None
            return
        candidates = [Path(typed), PROJECT_DATA_DIR / typed]
        dataset_path = next((c for c in candidates if c.is_file()), None)
        self.run_dir = run_folder_for(dataset_path) if dataset_path else None

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
        self.run_dir = run_folder_for(dataset)

        self.running = True
        self.run_button.config(state="disabled")
        self.status_var.set("Running analysis... this can take a few minutes.")
        self._set_log("")
        self.notebook.select(2)  # show the Run log tab while it runs

        thread = threading.Thread(
            target=self._run_pipeline, args=(dataset, self.run_dir), daemon=True
        )
        thread.start()

    def _run_pipeline(self, dataset, run_dir):
        env = dict(
            os.environ,
            PYTHONIOENCODING="utf-8",
            BATTERY_OUTPUT_ROOT=str(run_dir),
        )
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
        self._select_dataset_from_box()
        self._load_summary()
        self._load_report()
        self.show_chart()

    def _summary_file(self):
        return self.run_dir / "reports" / "battery_summary.csv" if self.run_dir else None

    def _report_file(self):
        return self.run_dir / "reports" / "battery_report.txt" if self.run_dir else None

    def _load_summary(self):
        self.summary_table.delete(*self.summary_table.get_children())
        summary = self._summary_file()
        if summary is None or not summary.exists():
            self.summary_table.insert(
                "", "end", values=("No results for this dataset yet", "Run an analysis")
            )
            return
        with summary.open(encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                self.summary_table.insert(
                    "", "end", values=(row.get("Metric", ""), _format_value(row.get("Value", "")))
                )

    def _load_report(self):
        self.report_text.config(state="normal")
        self.report_text.delete("1.0", "end")
        report = self._report_file()
        if report is not None and report.exists():
            self.report_text.insert("1.0", report.read_text(encoding="utf-8", errors="replace"))
        self.report_text.config(state="disabled")

    def show_chart(self):
        if self.run_dir is None:
            self.chart_label.config(image="", text="Choose a dataset to see its charts.")
            self.photo = None
            return
        path = charts_for(self.run_dir)[self.chart_var.get()]
        if not path.exists():
            self.chart_label.config(image="", text=f"No chart for this dataset:\n{path.name}")
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
        if self.run_dir is None:
            messagebox.showinfo("No dataset", "Choose a dataset first.")
            return
        self._open_path(charts_for(self.run_dir)[self.chart_var.get()])

    def open_outputs(self):
        self._open_path(self.run_dir if self.run_dir is not None else OUTPUT_DIR)

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
