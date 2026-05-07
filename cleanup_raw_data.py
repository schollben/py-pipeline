#!/usr/bin/env python3
"""
cleanup_raw_data.py — GUI tool to delete raw imaging data from processed BRUKER/SCANIMAGE folders.

A folder is "fully processed" when inference.h5 is present.

Files deleted from BRUKER folders:  *_Cycle*_Ch*_*.ome.tif + registered.h5
Files deleted from SCANIMAGE folders: file_NNNNN_NNNNN.tif + registered.h5
"""

import re
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, font, messagebox, ttk

BRUKER_RAW_TIFF = re.compile(r'.+_Cycle\d+_Ch\d+_\d+\.ome\.tif$')
SCANIMAGE_RAW_TIFF = re.compile(r'^file_\d{5}_\d{5}\.tif$')
REGISTERED_H5 = 'registered.h5'
PROCESSED_MARKER = 'inference.h5'


def fmt_size(n_bytes: int) -> str:
    for unit in ('B', 'KB', 'MB', 'GB', 'TB'):
        if n_bytes < 1024:
            return f'{n_bytes:.1f} {unit}'
        n_bytes /= 1024
    return f'{n_bytes:.1f} PB'


def classify_folder(folder: Path) -> str | None:
    for f in folder.iterdir():
        if not f.is_file():
            continue
        if BRUKER_RAW_TIFF.match(f.name):
            return 'BRUKER'
        if SCANIMAGE_RAW_TIFF.match(f.name):
            return 'SCANIMAGE'
    return None


def get_deletable_files(folder: Path, kind: str) -> list[Path]:
    candidates = []
    for f in folder.iterdir():
        if not f.is_file():
            continue
        if f.name == REGISTERED_H5:
            candidates.append(f)
        elif kind == 'BRUKER' and BRUKER_RAW_TIFF.match(f.name):
            candidates.append(f)
        elif kind == 'SCANIMAGE' and SCANIMAGE_RAW_TIFF.match(f.name):
            candidates.append(f)
    return sorted(candidates)


def scan_directory(root: Path) -> list[dict]:
    results = []
    for candidate in sorted(root.iterdir()):
        if not candidate.is_dir():
            continue
        kind = classify_folder(candidate)
        if kind is None:
            continue
        processed = (candidate / PROCESSED_MARKER).exists()
        deletable = get_deletable_files(candidate, kind) if processed else []
        results.append({
            'path': candidate,
            'kind': kind,
            'processed': processed,
            'files': deletable,
            'size': sum(f.stat().st_size for f in deletable),
        })
    return results


# ── GUI ───────────────────────────────────────────────────────────────────────

class CleanupApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title('Raw Data Cleanup')
        self.resizable(True, True)
        self._folder_vars: list[tuple[dict, tk.BooleanVar]] = []
        self._apply_dpi_scaling()
        self._build_ui()

    def _apply_dpi_scaling(self):
        self.tk.call('tk', 'scaling', 2.0)
        for fname in ('TkDefaultFont', 'TkTextFont', 'TkFixedFont',
                      'TkMenuFont', 'TkHeadingFont', 'TkCaptionFont'):
            try:
                font.nametofont(fname).configure(size=12)
            except Exception:
                pass
        self.minsize(900, 560)

    # ── layout ────────────────────────────────────────────────────────────────

    def _build_ui(self):
        pad = dict(padx=10, pady=6)

        # top bar: directory picker
        top = tk.Frame(self)
        top.pack(fill='x', **pad)

        tk.Label(top, text='Directory:').pack(side='left')
        self._dir_var = tk.StringVar()
        dir_entry = tk.Entry(top, textvariable=self._dir_var, width=55)
        dir_entry.pack(side='left', padx=(6, 4), fill='x', expand=True)
        tk.Button(top, text='Browse...', command=self._browse).pack(side='left')
        tk.Button(top, text='Scan', command=self._scan, width=8).pack(side='left', padx=(6, 0))

        # legend
        legend = tk.Frame(self)
        legend.pack(fill='x', padx=10, pady=(0, 2))
        tk.Label(legend, text='[+] processed', fg='#2a7a2a').pack(side='left', padx=(0, 12))
        tk.Label(legend, text='[-] not processed', fg='#888888').pack(side='left')

        # scrollable folder list
        list_frame = tk.Frame(self, bd=1, relief='sunken')
        list_frame.pack(fill='both', expand=True, padx=10, pady=(0, 4))

        self._canvas = tk.Canvas(list_frame, highlightthickness=0)
        sb = ttk.Scrollbar(list_frame, orient='vertical', command=self._canvas.yview)
        self._canvas.configure(yscrollcommand=sb.set)
        sb.pack(side='right', fill='y')
        self._canvas.pack(side='left', fill='both', expand=True)

        self._list_inner = tk.Frame(self._canvas)
        self._canvas_window = self._canvas.create_window((0, 0), window=self._list_inner, anchor='nw')
        self._list_inner.bind('<Configure>', self._on_inner_configure)
        self._canvas.bind('<Configure>', self._on_canvas_configure)
        self._canvas.bind_all('<MouseWheel>', self._on_mousewheel)
        self._canvas.bind_all('<Button-4>', self._on_mousewheel)
        self._canvas.bind_all('<Button-5>', self._on_mousewheel)

        # bottom bar
        bot = tk.Frame(self)
        bot.pack(fill='x', padx=10, pady=(0, 10))

        self._select_all_var = tk.BooleanVar(value=False)
        self._select_all_cb = tk.Checkbutton(
            bot, text='Select all', variable=self._select_all_var,
            command=self._toggle_all, state='disabled'
        )
        self._select_all_cb.pack(side='left')

        self._status_var = tk.StringVar(value='Choose a directory and click Scan.')
        tk.Label(bot, textvariable=self._status_var, anchor='w').pack(side='left', padx=12, fill='x', expand=True)

        self._delete_btn = tk.Button(
            bot, text='Delete Selected', command=self._confirm_delete,
            bg='#c0392b', fg='white', activebackground='#962d22',
            activeforeground='white', state='disabled', width=16
        )
        self._delete_btn.pack(side='right')

    # ── canvas helpers ────────────────────────────────────────────────────────

    def _on_inner_configure(self, *_):
        self._canvas.configure(scrollregion=self._canvas.bbox('all'))

    def _on_canvas_configure(self, event):
        self._canvas.itemconfig(self._canvas_window, width=event.width)

    def _on_mousewheel(self, event):
        if event.num == 4:
            self._canvas.yview_scroll(-1, 'units')
        elif event.num == 5:
            self._canvas.yview_scroll(1, 'units')
        else:
            self._canvas.yview_scroll(int(-event.delta / 120), 'units')

    # ── actions ───────────────────────────────────────────────────────────────

    def _browse(self):
        d = filedialog.askdirectory(title='Select imaging data directory')
        if d:
            self._dir_var.set(d)

    def _scan(self):
        root = Path(self._dir_var.get().strip())
        if not root.is_dir():
            messagebox.showerror('Error', f'Not a valid directory:\n{root}')
            return

        self._clear_list()
        self._status_var.set('Scanning...')
        self.update_idletasks()

        results = scan_directory(root)

        if not results:
            self._status_var.set('No BRUKER or SCANIMAGE folders found.')
            return

        self._folder_vars.clear()
        for r in results:
            var = tk.BooleanVar(value=r['processed'] and bool(r['files']))
            self._folder_vars.append((r, var))
            self._add_row(r, var)

        self._select_all_cb.config(state='normal')
        self._delete_btn.config(state='normal')
        self._update_status()

    def _add_row(self, r: dict, var: tk.BooleanVar):
        row = tk.Frame(self._list_inner, bd=0)
        row.pack(fill='x', padx=4, pady=2)

        can_delete = r['processed'] and bool(r['files'])
        color = '#2a7a2a' if r['processed'] else '#888888'

        cb = tk.Checkbutton(
            row, variable=var,
            state='normal' if can_delete else 'disabled',
            command=self._update_status
        )
        cb.pack(side='left')

        kind_lbl = tk.Label(row, text=f'[{r["kind"]}]', width=10, anchor='w', fg='#444')
        kind_lbl.pack(side='left')

        name_lbl = tk.Label(row, text=r['path'].name, anchor='w', fg=color, font=('TkDefaultFont', 10, 'bold'))
        name_lbl.pack(side='left', fill='x', expand=True)

        if r['processed'] and r['files']:
            size_lbl = tk.Label(row, text=fmt_size(r['size']), width=10, anchor='e', fg='#c0392b')
            size_lbl.pack(side='right')
            count_lbl = tk.Label(row, text=f'{len(r["files"])} file(s)', width=12, anchor='e', fg='#555')
            count_lbl.pack(side='right')
        elif r['processed'] and not r['files']:
            tk.Label(row, text='already clean', fg='#2a7a2a', width=14, anchor='e').pack(side='right')
        else:
            tk.Label(row, text='not processed', fg='#888888', width=14, anchor='e').pack(side='right')

        # separator
        ttk.Separator(self._list_inner, orient='horizontal').pack(fill='x', padx=4)

    def _clear_list(self):
        for widget in self._list_inner.winfo_children():
            widget.destroy()
        self._folder_vars.clear()
        self._select_all_cb.config(state='disabled')
        self._delete_btn.config(state='disabled')
        self._select_all_var.set(False)

    def _toggle_all(self):
        state = self._select_all_var.get()
        for r, var in self._folder_vars:
            if r['processed'] and r['files']:
                var.set(state)
        self._update_status()

    def _update_status(self):
        selected = [(r, var) for r, var in self._folder_vars if var.get()]
        total_size = sum(r['size'] for r, _ in selected)
        total_files = sum(len(r['files']) for r, _ in selected)
        if selected:
            self._status_var.set(
                f'{len(selected)} folder(s) selected | {total_files} file(s) | {fmt_size(total_size)} to free'
            )
        else:
            self._status_var.set('No folders selected.')

    def _confirm_delete(self):
        selected = [(r, var) for r, var in self._folder_vars if var.get()]
        if not selected:
            messagebox.showinfo('Nothing selected', 'Check at least one folder to delete.')
            return

        total_size = sum(r['size'] for r, _ in selected)
        total_files = sum(len(r['files']) for r, _ in selected)
        names = '\n'.join(f'  - {r["path"].name}' for r, _ in selected)

        ok = messagebox.askyesno(
            'Confirm deletion',
            f'Permanently delete {total_files} file(s) ({fmt_size(total_size)}) from:\n\n{names}\n\nThis cannot be undone.'
        )
        if not ok:
            return

        deleted = errors = 0
        for r, _ in selected:
            for f in r['files']:
                try:
                    f.unlink()
                    deleted += 1
                except Exception as e:
                    messagebox.showerror('Error', f'Could not delete:\n{f}\n\n{e}')
                    errors += 1

        messagebox.showinfo('Done', f'Deleted {deleted} file(s).\nErrors: {errors}')
        self._scan()  # refresh


if __name__ == '__main__':
    app = CleanupApp()
    app.mainloop()
