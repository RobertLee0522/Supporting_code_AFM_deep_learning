"""
AFM 盲探針重建工具 v4
修正：自動計算合理 max_sz / half / 偵測閾值
依賴：pip install numpy scipy matplotlib Pillow
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import threading, os, re
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from scipy.ndimage import label, center_of_mass, shift
from scipy.io import savemat
from scipy.signal import correlate2d

C = {
    'bg':'#0f1117','panel':'#1a1d27','border':'#2a2d3a',
    'accent':'#4f9cf9','success':'#22c55e','warn':'#f59e0b','danger':'#ef4444',
    'text':'#e2e8f0','muted':'#64748b','input_bg':'#252836','hover':'#2d3348',
}

# ══════════════════════════════════════════════════════════════════
# Float Entry
# ══════════════════════════════════════════════════════════════════
def make_float_entry(parent, textvariable, **kwargs):
    def validate(val):
        if val in ('', '-', '.', '-.'): return True
        try: float(val); return True
        except ValueError:
            return val.count('.') <= 1 and all(c in '0123456789.-' for c in val)
    vcmd = (parent.register(validate), '%P')
    return tk.Entry(parent, textvariable=textvariable,
                    validate='key', validatecommand=vcmd,
                    bg=C['input_bg'], fg=C['text'],
                    insertbackground=C['text'], relief='flat',
                    font=('Helvetica', 10), bd=0, **kwargs)

# ══════════════════════════════════════════════════════════════════
# Nanoscope 讀取
# ══════════════════════════════════════════════════════════════════
def parse_nanoscope(fp):
    with open(fp,'rb') as f: raw = f.read(65536)
    hdr = raw.decode('latin-1')
    def fv(pat, cast=str, default=None):
        m = re.search(pat, hdr); return cast(m.group(1)) if m else default
    n_px = fv(r'\\Samps/line:\s*(\d+)', int, 256)
    n_ln = fv(r'\\(?:Number of lines|Lines):\s*(\d+)', int, 256)
    m = re.search(r'\\Scan Size:\s*([\d.]+)\s*([\d.]*)\s*(~m|nm|um)', hdr)
    scan = float(m.group(1))*1000 if m and m.group(3) in ('~m','um') \
           else (float(m.group(1)) if m else 5000.)
    zss = fv(r'@Sens\. ZsensSens:\s*V\s*([\d.]+)', float, 813.1653)
    zs  = fv(r'@Sens\. Zsens:\s*V\s*([\d.]+)',     float, 32.07862)
    chs = []
    for blk in hdr.split('\\*Ciao image list'):
        if not blk.strip(): continue
        off = re.search(r'\\Data offset:\s*(\d+)', blk)
        bpp = re.search(r'\\Bytes/pixel:\s*(\d+)', blk)
        nm  = re.search(r'@2:Image Data:\s*S\s*\[([^\]]+)\].*?"([^"]+)"', blk)
        zsc = re.search(r'@2:Z scale:\s*V \[Sens\.\s*(\w+)\]\s*\(([\d.e+-]+)', blk)
        if off and nm:
            chs.append({'offset':int(off.group(1)), 'bpp':int(bpp.group(1)) if bpp else 2,
                        'key':nm.group(1).strip(), 'name':nm.group(2).strip(),
                        'z_key':zsc.group(1) if zsc else 'ZsensSens',
                        'z_lsb':float(zsc.group(2)) if zsc else 0.000375})
    return {'n_px':n_px,'n_lines':n_ln,'scan_nm':scan,'px_nm':scan/n_px,
            'zsens':zs,'zsens_s':zss,'channels':chs,'filepath':fp}

def read_channel(fp, meta, idx=0):
    ch = meta['channels'][idx]; n = meta['n_px']*meta['n_lines']
    dt = np.int16 if ch['bpp']==2 else np.int32
    with open(fp,'rb') as f:
        f.seek(ch['offset']); raw = np.frombuffer(f.read(n*ch['bpp']), dtype=dt)
    img = raw.reshape(meta['n_lines'], meta['n_px']).astype(np.float64)
    sens = meta['zsens_s'] if 'ZsensSens' in ch['z_key'] else meta['zsens']
    return img * ch['z_lsb'] * sens

def flatten_rows(img):
    flat = img.copy(); x = np.arange(img.shape[1])
    for i in range(img.shape[0]):
        c = np.polyfit(x, flat[i], 1); flat[i] -= np.polyval(c, x)
    return flat - np.median(flat)

def auto_height_ch(meta):
    for p in ['ZSensor','Height Sensor','Height']:
        for i,ch in enumerate(meta['channels']):
            if p.lower() in ch['name'].lower(): return i
    return 0

# ══════════════════════════════════════════════════════════════════
# 自動計算合理參數（核心修正）
# ══════════════════════════════════════════════════════════════════
def auto_params(geom, shape_type, meta):
    """
    根據幾何與掃描參數，自動推算：
    - half     : patch 半徑（px）
    - max_sz   : 最大連通區域面積（px²）
    - pct_hi/lo: 偵測閾值百分位數
    - warn_msgs: 警告訊息列表
    """
    px = meta['px_nm']
    n  = meta['n_px']
    sc = meta['scan_nm']
    p  = geom['pitch']

    # 特徵最大半徑（頂部 or 底部取較大者）
    r_max_nm = max(geom['d_bot'], geom['d_top']) / 2
    r_max_px = r_max_nm / px

    # half：特徵半徑 + 20% buffer，至少比特徵半徑大 5px
    half = max(int(r_max_px * 1.2) + 5, 15)
    # 不能超過影像的 1/4
    half = min(half, n // 4 - 2)

    # max_sz：特徵頂面面積的 3 倍（容納誤差）
    feat_area = np.pi * r_max_px**2
    max_sz = int(feat_area * 3) + 100
    max_sz = max(max_sz, 500)

    # 偵測閾值：依特徵相對面積調整
    feat_fill = feat_area / (n * n)      # 特徵占影像比例
    # 對梯形柱（高點），用高百分位數閾值
    # 若特徵很大（>10%），把閾值調高
    if shape_type == 'cylinder_hole':
        pct = max(5, min(15, int(feat_fill * 200)))
        pct_lo, pct_hi = pct, None
    else:
        pct = max(70, min(95, 100 - int(feat_fill * 150)))
        pct_lo, pct_hi = None, pct

    warnings = []
    if p > sc:
        warnings.append(f"⚠ 週期({p}nm) > 掃描範圍({sc:.0f}nm)，視野內約 {sc/p:.1f} 個特徵")
    if half >= n//4:
        warnings.append(f"⚠ Patch 半徑({half}px)已達上限，建議縮小掃描範圍或降低 pitch")
    if feat_fill > 0.3:
        warnings.append(f"⚠ 特徵占影像 {feat_fill*100:.0f}%，建議用更大掃描範圍")

    return {'half': half, 'max_sz': max_sz,
            'pct_lo': pct_lo, 'pct_hi': pct_hi,
            'r_max_px': r_max_px, 'feat_area': feat_area,
            'warnings': warnings}

# ══════════════════════════════════════════════════════════════════
# GT 生成
# ══════════════════════════════════════════════════════════════════
def make_gt(n_px, scan_nm, geom, shape_type):
    px = scan_nm / n_px
    p  = geom['pitch'] / px
    h  = geom['height']
    gt = np.zeros((n_px, n_px)) if shape_type == 'trapezoid_pillar' \
         else np.full((n_px, n_px), h)
    r_bot = (geom['d_bot']/2) / px
    r_top = (geom['d_top']/2) / px
    yi, xi = np.indices((n_px, n_px))
    for cy in np.arange(-p/2, n_px+p, p):
        for cx in np.arange(-p/2, n_px+p, p):
            dist = np.sqrt((xi-cx)**2+(yi-cy)**2)
            if shape_type == 'cylinder_hole':
                gt[dist <= r_bot] = 0.
            else:
                r_min, r_max = min(r_bot,r_top), max(r_bot,r_top)
                gt[dist <= r_top] = np.maximum(gt[dist <= r_top], h)
                side = (dist > r_min) & (dist <= r_max)
                if r_max > r_min:
                    sl = h*(r_max - dist[side])/(r_max - r_min)
                    np.maximum(gt[side], sl, out=gt[side])
    return gt

def make_gt_patch(geom, shape_type, half, px_nm):
    size = 2*half; h = geom['height']
    gt = np.zeros((size,size))
    yi,xi = np.indices((size,size))
    dist = np.sqrt((xi-half)**2+(yi-half)**2)
    r_bot = (geom['d_bot']/2)/px_nm
    r_top = (geom['d_top']/2)/px_nm
    if shape_type == 'cylinder_hole':
        gt[dist <= r_bot] = -h
    else:
        r_min, r_max = min(r_bot,r_top), max(r_bot,r_top)
        gt[dist <= r_top] = h
        side = (dist > r_min) & (dist <= r_max)
        if r_max > r_min:
            gt[side] = h*(r_max - dist[side])/(r_max - r_min)
    return gt

def align_gt(scan, gt, shape_type):
    if shape_type == 'cylinder_hole':
        sb = (scan < np.percentile(scan,15)).astype(float)
        gb = (gt < gt.max()*0.1).astype(float)
    else:
        sb = (scan > np.percentile(scan,80)).astype(float)
        gb = (gt > gt.max()*0.5).astype(float)
    c = correlate2d(sb, gb, mode='same')
    pk = np.unravel_index(c.argmax(), c.shape)
    dy, dx = pk[0]-c.shape[0]//2, pk[1]-c.shape[1]//2
    return shift(gt,[dy,dx],mode='wrap'), dy, dx

# ══════════════════════════════════════════════════════════════════
# 偵測（max_sz 動態傳入）
# ══════════════════════════════════════════════════════════════════
def detect_features(scan, shape_type, pct_lo, pct_hi, mn=5, mx=65536):
    if shape_type == 'cylinder_hole':
        thr = np.percentile(scan, pct_lo); mask = scan < thr
    else:
        thr = np.percentile(scan, pct_hi); mask = scan > thr
    lbl, n = label(mask)
    sz  = [(lbl==i).sum() for i in range(1,n+1)]
    ids = [i+1 for i,s in enumerate(sz) if mn<=s<=mx]
    return [(int(round(y)),int(round(x)))
            for y,x in center_of_mass(mask,lbl,ids)], n, len(ids)

def reconstruct(scan, geom, shape_type, meta, ap):
    half   = ap['half']
    max_sz = ap['max_sz']
    pct_lo = ap['pct_lo'] if ap['pct_lo'] else 8
    pct_hi = ap['pct_hi'] if ap['pct_hi'] else 92

    gt = make_gt(meta['n_px'], meta['scan_nm'], geom, shape_type)
    gt_al, dy, dx = align_gt(scan, gt, shape_type)

    centers, n_raw, n_valid = detect_features(
        scan, shape_type, pct_lo, pct_hi, mn=5, mx=max_sz)

    # reflect padding：特徵靠近邊緣時仍可取完整 patch
    padded = np.pad(scan, half, mode='reflect')
    n = scan.shape[0]; patches = []; edge_cnt = 0
    for cy,cx in centers:
        cy_p, cx_p = cy + half, cx + half
        p = padded[cy_p-half:cy_p+half, cx_p-half:cx_p+half]
        if p.shape == (2*half, 2*half):
            patches.append(p)
            if min(cy, cx, n-1-cy, n-1-cx) < half:
                edge_cnt += 1

    edge_note = f' (含{edge_cnt}個邊緣特徵，用padding補全)' if edge_cnt else ''
    info = (f'偵測: {n_raw}個候選 → 過濾後{n_valid}個 → patch收集{len(patches)}個{edge_note}\n'
            f'  GT對齊: dy={dy}px dx={dx}px | half={half}px | max_sz={max_sz}px²')

    if not patches: return None,None,None,0,gt_al,info

    avg = np.mean(patches, axis=0)
    if shape_type == 'cylinder_hole': avg -= avg.max()
    else: avg -= avg.min()

    gt_p = make_gt_patch(geom, shape_type, half, meta['px_nm'])
    tip  = avg - gt_p
    tip -= tip.max()
    return tip, avg, gt_p, len(patches), gt_al, info

# ══════════════════════════════════════════════════════════════════
# GUI
# ══════════════════════════════════════════════════════════════════
class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title('AFM Blind Tip Reconstruction  v4')
        self.configure(bg=C['bg']); self.geometry('1380x900'); self.minsize(1150,740)
        self.meta=None; self.scan=None; self.tip=None
        self.afm_path = tk.StringVar(value='尚未選擇')
        self.ch_var   = tk.StringVar()
        self.shape_var= tk.StringVar(value='trapezoid_pillar')
        self.d_bot    = tk.StringVar(value='183.5')
        self.d_top    = tk.StringVar(value='228.8')
        self.height   = tk.StringVar(value='125.8')
        self.pitch    = tk.StringVar(value='1000')
        # 進階（可覆蓋自動值）
        self.half_v   = tk.StringVar(value='auto')
        self.max_sz_v = tk.StringVar(value='auto')
        self.pct_v    = tk.StringVar(value='auto')
        for v in (self.d_bot,self.d_top,self.height):
            v.trace_add('write', lambda *_: self.after(80, self._update_info))
        self.fig_history = []   # [(label, Figure), ...]
        self._build(); self.protocol('WM_DELETE_WINDOW', self.on_close)

    # ── 佈局 ──────────────────────────────────────────────────────
    def _build(self):
        self.columnconfigure(0,weight=0); self.columnconfigure(1,weight=1)
        self.rowconfigure(0,weight=1)
        self._sidebar(); self._main_area()

    def _sidebar(self):
        sb = tk.Frame(self, bg=C['panel'], width=330)
        sb.grid(row=0,column=0,sticky='nsew'); sb.grid_propagate(False)
        sb.columnconfigure(0,weight=1)

        tk.Label(sb,text='AFM Tip',font=('Helvetica',20,'bold'),
                 bg=C['panel'],fg=C['accent']).pack(pady=(16,0))
        tk.Label(sb,text='Reconstruction  v4',font=('Helvetica',10),
                 bg=C['panel'],fg=C['muted']).pack(pady=(0,10))
        ttk.Separator(sb).pack(fill='x',padx=14)

        # ① AFM
        self._sec(sb,'① 選擇 AFM 檔案')
        pf=tk.Frame(sb,bg=C['panel']); pf.pack(fill='x',padx=14,pady=(0,6))
        pf.columnconfigure(0,weight=1)
        tk.Label(pf,textvariable=self.afm_path,bg=C['input_bg'],fg=C['text'],
                 font=('Courier',8),wraplength=210,anchor='w',padx=6,pady=5
                 ).grid(row=0,column=0,sticky='ew',padx=(0,6))
        self._btn(pf,'瀏覽',self.browse,small=True).grid(row=0,column=1)
        tk.Label(sb,text='高度通道',bg=C['panel'],fg=C['muted'],font=('Helvetica',9)).pack(anchor='w',padx=14)
        self.ch_cb=ttk.Combobox(sb,textvariable=self.ch_var,state='readonly',font=('Helvetica',9))
        self.ch_cb.pack(fill='x',padx=14,pady=(2,10))
        self.ch_cb.bind('<<ComboboxSelected>>',lambda _:self._preview())
        ttk.Separator(sb).pack(fill='x',padx=14)

        # ② 形狀
        self._sec(sb,'② 特徵形狀')
        sf=tk.Frame(sb,bg=C['panel']); sf.pack(fill='x',padx=14,pady=(0,6))
        for txt,val in [('梯形柱 Trapezoid Pillar','trapezoid_pillar'),
                        ('圓柱孔 Cylinder Hole',   'cylinder_hole')]:
            tk.Radiobutton(sf,text=txt,variable=self.shape_var,value=val,
                           bg=C['panel'],fg=C['text'],selectcolor=C['input_bg'],
                           activebackground=C['panel'],font=('Helvetica',9),
                           command=self._on_shape).pack(anchor='w',pady=2)
        ttk.Separator(sb).pack(fill='x',padx=14)

        # ③ 幾何
        self._sec(sb,'③ SEM 幾何參數')
        self.geom_frame=tk.Frame(sb,bg=C['panel'])
        self.geom_frame.pack(fill='x')
        self._rebuild_geom()
        ttk.Separator(sb).pack(fill='x',padx=14)

        # ④ 進階（顯示自動計算值，可手動覆蓋）
        self._sec(sb,'④ 進階設定（auto = 自動計算）')
        self._row(sb,'Patch 半徑 (px)',   self.half_v)
        self._row(sb,'最大特徵面積 (px²)', self.max_sz_v)
        self._row(sb,'偵測閾值 (%)',       self.pct_v)

        # 自動參數提示框
        self.auto_info = tk.Label(sb,text='',bg=C['panel'],fg=C['warn'],
                                  font=('Courier',8),wraplength=295,justify='left')
        self.auto_info.pack(anchor='w',padx=14,pady=(4,0))

        tk.Frame(sb,bg=C['panel'],height=8).pack()
        self._btn(sb,'▶  執行重建',self.run,big=True).pack(fill='x',padx=14,pady=3)
        self._btn(sb,'💾  儲存 tip.mat',self.save,color=C['success']
                  ).pack(fill='x',padx=14,pady=3)

        ttk.Separator(sb).pack(fill='x',padx=14)
        tk.Label(sb,text='LOG',bg=C['panel'],fg=C['muted'],font=('Helvetica',8)
                 ).pack(anchor='w',padx=14,pady=(6,2))
        lf=tk.Frame(sb,bg=C['input_bg'])
        lf.pack(fill='both',expand=True,padx=14,pady=(0,14))
        self.log_box=tk.Text(lf,bg=C['input_bg'],fg='#94a3b8',font=('Courier',8),
                             relief='flat',state='disabled',wrap='word')
        sc=ttk.Scrollbar(lf,command=self.log_box.yview)
        self.log_box.configure(yscrollcommand=sc.set)
        sc.pack(side='right',fill='y'); self.log_box.pack(fill='both',expand=True,padx=4,pady=4)

    def _rebuild_geom(self):
        for w in self.geom_frame.winfo_children(): w.destroy()
        s=self.shape_var.get()
        if s=='trapezoid_pillar':
            self.angle_lbl=tk.Label(self.geom_frame,text='',bg=C['panel'],fg=C['warn'],
                                    font=('Helvetica',9)); self.angle_lbl.pack(anchor='w',padx=14)
            fields=[('底部寬 d_bot (nm)',self.d_bot),
                    ('頂部寬 d_top (nm)',self.d_top),
                    ('高度 height  (nm)',self.height),
                    ('週期 pitch   (nm)',self.pitch)]
        else:
            self.angle_lbl=None
            fields=[('孔洞直徑 d  (nm)',self.d_bot),
                    ('孔洞深度 z  (nm)',self.height),
                    ('週期 pitch  (nm)',self.pitch)]
        for lbl,var in fields: self._row(self.geom_frame,lbl,var)
        self._update_info()

    def _on_shape(self): self._rebuild_geom()

    def _update_info(self):
        # 更新側壁角
        if hasattr(self,'angle_lbl') and self.angle_lbl:
            try:
                bot=float(self.d_bot.get() or 0); top=float(self.d_top.get() or 0)
                h=float(self.height.get() or 0)
                if h>0 and abs(top-bot)>0:
                    diff=abs(top-bot)/2; taper=np.degrees(np.arctan(diff/h))
                    kind='倒梯形(底窄頂寬)' if top>bot else '正梯形(底寬頂窄)'
                    self.angle_lbl.config(text=f'  {kind}  taper: {taper:.1f}°')
            except: pass
        # 更新自動參數提示
        if self.meta:
            try:
                geom=self._get_geom()
                ap=auto_params(geom,self.shape_var.get(),self.meta)
                info=(f'auto: half={ap["half"]}px  max_sz={ap["max_sz"]}px²\n'
                      f'      pct={ap["pct_hi"] or ap["pct_lo"]}%  '
                      f'r_feat={ap["r_max_px"]:.1f}px')
                if ap['warnings']:
                    info += '\n' + '\n'.join(ap['warnings'])
                self.auto_info.config(text=info)
            except: pass

    def _main_area(self):
        m=tk.Frame(self,bg=C['bg']); m.grid(row=0,column=1,sticky='nsew',padx=14,pady=14)
        m.columnconfigure(0,weight=1); m.rowconfigure(1,weight=1)
        sb=tk.Frame(m,bg=C['panel'],height=44); sb.grid(row=0,column=0,sticky='ew',pady=(0,10))
        self.stats={}
        for i,(k,v) in enumerate([('AFM','—'),('掃描範圍','—'),('px_nm','—'),('Ra','—'),('特徵數','—')]):
            f=tk.Frame(sb,bg=C['panel']); f.grid(row=0,column=i,padx=16,pady=8)
            tk.Label(f,text=k,bg=C['panel'],fg=C['muted'],font=('Helvetica',8)).pack()
            l=tk.Label(f,text=v,bg=C['panel'],fg=C['text'],font=('Helvetica',10,'bold'))
            l.pack(); self.stats[k]=l
        self.fig=Figure(figsize=(10,6.2),facecolor=C['bg'])
        self.canvas=FigureCanvasTkAgg(self.fig,master=m)
        self.canvas.get_tk_widget().grid(row=1,column=0,sticky='nsew')

        # 圖片歷史工具列
        hist_bar = tk.Frame(m, bg=C['panel'], height=38)
        hist_bar.grid(row=2, column=0, sticky='ew', pady=(8,0))
        hist_bar.columnconfigure(1, weight=1)
        tk.Label(hist_bar, text='已保留圖片：', bg=C['panel'], fg=C['muted'],
                 font=('Helvetica',9)).grid(row=0,column=0,padx=10,pady=8)
        self.hist_frame = tk.Frame(hist_bar, bg=C['panel'])
        self.hist_frame.grid(row=0, column=1, sticky='ew', pady=6)
        self._btn(hist_bar, '📷 保留目前圖片', self.keep_fig,
                  color=C['accent2']).grid(row=0, column=2, padx=10, pady=4)
        self._placeholder()

    def _sec(self,p,t):
        tk.Label(p,text=t,bg=C['panel'],fg=C['accent'],
                 font=('Helvetica',10,'bold')).pack(anchor='w',padx=14,pady=(9,3))

    def _row(self,p,label,var):
        f=tk.Frame(p,bg=C['panel']); f.pack(fill='x',padx=14,pady=2); f.columnconfigure(1,weight=1)
        tk.Label(f,text=label,bg=C['panel'],fg=C['muted'],font=('Helvetica',9),
                 width=19,anchor='w').grid(row=0,column=0)
        make_float_entry(f,var).grid(row=0,column=1,sticky='ew',ipady=4,padx=(6,0))

    def _btn(self,p,t,cmd,big=False,small=False,color=None):
        c=color or C['accent']
        return tk.Button(p,text=t,command=cmd,bg=c,fg='white',
                         activebackground=C['hover'],activeforeground='white',
                         relief='flat',cursor='hand2',
                         font=('Helvetica',11 if big else (9 if small else 10),
                               'bold' if big else 'normal'),pady=8 if big else 4)

    def _placeholder(self):
        self.fig.clear(); ax=self.fig.add_subplot(111); ax.set_facecolor(C['panel'])
        ax.text(0.5,0.5,'選擇 AFM 檔案後開始分析',ha='center',va='center',
                color=C['muted'],fontsize=14,transform=ax.transAxes)
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values(): sp.set_edgecolor(C['border'])
        self.canvas.draw()

    def log(self,msg):
        self.log_box.configure(state='normal')
        self.log_box.insert('end',msg+'\n'); self.log_box.see('end')
        self.log_box.configure(state='disabled')

    def _get_geom(self):
        s=self.shape_var.get()
        return {'d_bot':float(self.d_bot.get()),
                'd_top':float(self.d_top.get()) if s=='trapezoid_pillar' else float(self.d_bot.get()),
                'height':float(self.height.get()),
                'pitch':float(self.pitch.get())}

    def browse(self):
        p=filedialog.askopenfilename(title='選擇 Nanoscope 檔案',
            filetypes=[('Nanoscope','*.000 *.001 *.002 *.003 *.004 *.005'),('All','*.*')])
        if not p: return
        try:
            self.meta=parse_nanoscope(p)
            self.afm_path.set(os.path.basename(p))
            names=[f"[{i}] {ch['name']}" for i,ch in enumerate(self.meta['channels'])]
            self.ch_cb['values']=names; auto=auto_height_ch(self.meta)
            self.ch_cb.current(auto); self.ch_var.set(names[auto])
            self._preview(); self._update_info()
            self.log(f'✓ {os.path.basename(p)} | {len(self.meta["channels"])}通道 | '
                     f'{self.meta["scan_nm"]:.1f}nm | {self.meta["n_px"]}px')
        except Exception as e:
            messagebox.showerror('讀取失敗',str(e)); self.log(f'✗ {e}')

    def _preview(self):
        if not self.meta: return
        raw=read_channel(self.meta['filepath'],self.meta,self.ch_cb.current())
        self.scan=flatten_rows(raw)
        m=self.meta; s=self.scan
        self.stats['AFM'].config(text=os.path.basename(m['filepath']))
        self.stats['掃描範圍'].config(text=f'{m["scan_nm"]:.1f}nm')
        self.stats['px_nm'].config(text=f'{m["px_nm"]:.2f}nm/px')
        self.stats['Ra'].config(text=f'{np.mean(np.abs(s-s.mean())):.2f}nm')
        self._draw_preview()

    def _draw_preview(self):
        self.fig.clear(); self.fig.patch.set_facecolor(C['bg'])
        gs=self.fig.add_gridspec(1,2,wspace=0.3,left=0.06,right=0.97,top=0.9,bottom=0.1)
        s=self.scan; m=self.meta; ext=[0,m['scan_nm'],m['scan_nm'],0]
        def da(ax,t):
            ax.set_facecolor(C['panel']); ax.set_title(t,color=C['text'],fontsize=10)
            ax.tick_params(colors=C['muted'],labelsize=7)
            for sp in ax.spines.values(): sp.set_edgecolor(C['border'])
        ax1=self.fig.add_subplot(gs[0]); da(ax1,'AFM Scan (Height Sensor)')
        im=ax1.imshow(s,cmap='afmhot',origin='upper',extent=ext,
                      vmin=np.percentile(s,2),vmax=np.percentile(s,98))
        cb=self.fig.colorbar(im,ax=ax1,fraction=0.046); cb.ax.tick_params(colors='#666',labelsize=6)
        ax1.set_xlabel('nm',color=C['muted'],fontsize=8); ax1.set_ylabel('nm',color=C['muted'],fontsize=8)
        ax2=self.fig.add_subplot(gs[1]); da(ax2,'Line Profiles')
        x=np.linspace(0,m['scan_nm'],s.shape[1])
        for row,col in zip([s.shape[0]//4,s.shape[0]//2,3*s.shape[0]//4],
                           ['#4f9cf9','#f59e0b','#ef4444']):
            ax2.plot(x,s[row,:],color=col,lw=1.2,alpha=0.8,label=f'row {row}')
        ax2.set_xlabel('nm',color=C['muted'],fontsize=8); ax2.set_ylabel('nm',color=C['muted'],fontsize=8)
        ax2.grid(color=C['border'],alpha=0.5)
        leg=ax2.legend(fontsize=7,facecolor=C['panel'],edgecolor=C['border'])
        [t.set_color(C['text']) for t in leg.get_texts()]
        self.canvas.draw()

    def run(self):
        if self.scan is None:
            messagebox.showwarning('提示','請先選擇 AFM 檔案'); return
        try:
            geom=self._get_geom()
            s=self.shape_var.get()
            ap=auto_params(geom,s,self.meta)
            # 允許手動覆蓋
            if self.half_v.get() not in ('','auto'):
                ap['half']=int(float(self.half_v.get()))
            if self.max_sz_v.get() not in ('','auto'):
                ap['max_sz']=int(float(self.max_sz_v.get()))
            if self.pct_v.get() not in ('','auto'):
                pct=float(self.pct_v.get())
                if s=='cylinder_hole': ap['pct_lo']=pct
                else: ap['pct_hi']=pct
        except ValueError:
            messagebox.showerror('錯誤','請確認所有參數為有效數字'); return

        self.log(f'執行重建 shape={s}')
        self.log(f'  d_bot={geom["d_bot"]} d_top={geom["d_top"]} h={geom["height"]} p={geom["pitch"]}')
        self.log(f'  half={ap["half"]}px  max_sz={ap["max_sz"]}px²  pct={ap["pct_hi"] or ap["pct_lo"]}%')
        for w in ap['warnings']: self.log(f'  {w}')

        def worker():
            try:
                tip,avg,gt_p,n,gt_al,info=reconstruct(self.scan,geom,s,self.meta,ap)
                self.log(info)
                if tip is None:
                    self.after(0,lambda:messagebox.showerror('失敗',
                        '找不到有效特徵\n\n建議：\n1. 確認形狀選擇正確\n2. 調整偵測閾值\n3. 縮小掃描範圍讓特徵更明顯'))
                    self.log('✗ 找不到有效特徵'); return
                self.tip=tip; self.tip_half=ap['half']; self.geom=geom; self.shape_type=s
                px=self.meta['px_nm']; mid=ap['half']; tip1d=tip[mid,:]
                beyond=np.where(tip1d < tip.min()*0.1)[0]
                r_est=(beyond[-1]-beyond[0])*px/2 if len(beyond)>1 else float('nan')
                feat_h=abs(avg.min() if s=='cylinder_hole' else avg.max())
                self.log(f'✓ {n}個特徵平均 | 量測高度:{feat_h:.1f}nm (GT:{geom["height"]}nm)')
                if not np.isnan(r_est): self.log(f'  估算探針半徑:{r_est:.1f}nm')
                self.after(0,lambda:self._draw_result(tip,avg,gt_p,gt_al,geom,s,n,r_est,ap['half'],px))
                self.after(0,lambda:self.stats['特徵數'].config(text=str(n)))
            except Exception as e:
                self.log(f'✗ {e}'); import traceback; traceback.print_exc()

        threading.Thread(target=worker,daemon=True).start()

    def _draw_result(self,tip,avg,gt_p,gt_al,geom,shape,n,r_est,half,px):
        self.fig.clear(); self.fig.patch.set_facecolor(C['bg'])
        gs=self.fig.add_gridspec(2,3,hspace=0.42,wspace=0.32,
                                 left=0.06,right=0.97,top=0.92,bottom=0.07)
        m=self.meta; ext=[0,m['scan_nm'],m['scan_nm'],0]; s=self.scan
        def da(ax,t):
            ax.set_facecolor(C['panel']); ax.set_title(t,color=C['text'],fontsize=9,pad=4)
            ax.tick_params(colors=C['muted'],labelsize=6)
            for sp in ax.spines.values(): sp.set_edgecolor(C['border'])
        def dcb(im,ax):
            c=self.fig.colorbar(im,ax=ax,fraction=0.046); c.ax.tick_params(colors='#666',labelsize=5)

        ax1=self.fig.add_subplot(gs[0,0]); da(ax1,'AFM Scan')
        im=ax1.imshow(s,cmap='afmhot',origin='upper',extent=ext,
                      vmin=np.percentile(s,2),vmax=np.percentile(s,98)); dcb(im,ax1)
        ax1.set_xlabel('nm',color=C['muted'],fontsize=7)

        lbl=(f'd_bot={geom["d_bot"]:.1f} d_top={geom["d_top"]:.1f} h={geom["height"]:.1f}nm'
             if shape=='trapezoid_pillar' else f'd={geom["d_bot"]:.1f} z={geom["height"]:.1f}nm')
        ax2=self.fig.add_subplot(gs[0,1]); da(ax2,f'Synthetic GT  {lbl}')
        im2=ax2.imshow(gt_al,cmap='afmhot',origin='upper',extent=ext); dcb(im2,ax2)

        avg_d=avg.copy()
        if shape=='cylinder_hole': avg_d-=avg_d.max()
        else: avg_d-=avg_d.min()
        ax3=self.fig.add_subplot(gs[0,2]); da(ax3,f'Avg AFM Feature  (n={n})')
        im3=ax3.imshow(avg_d,cmap='afmhot',origin='upper'); dcb(im3,ax3)
        ax3.axhline(half,color='#f00',ls='--',lw=0.6,alpha=0.6)
        ax3.axvline(half,color='#f00',ls='--',lw=0.6,alpha=0.6)

        ax4=self.fig.add_subplot(gs[1,0]); da(ax4,'Estimated Tip Shape')
        im4=ax4.imshow(tip,cmap='RdBu_r',origin='upper',vmin=tip.min(),vmax=0); dcb(im4,ax4)
        ax4.axhline(half,color='w',ls='--',lw=0.5,alpha=0.4)
        ax4.axvline(half,color='w',ls='--',lw=0.5,alpha=0.4)

        ax5=self.fig.add_subplot(gs[1,1]); da(ax5,'Cross-section Profile')
        x=(np.arange(2*half)-half)*px
        ax5.plot(x,avg_d[half,:],color='#4f9cf9',lw=1.8,label='AFM feature')
        ax5.plot(x,gt_p[half,:], color='#ef4444',lw=1.5,ls='--',label='GT ideal')
        ax5.plot(x,tip[half,:],  color='#22c55e',lw=2,  label='Tip')
        ax5.axvline(0,color=C['border'],ls=':',lw=1)
        ax5.set_xlabel('nm',color=C['muted'],fontsize=7); ax5.set_ylabel('nm',color=C['muted'],fontsize=7)
        leg=ax5.legend(fontsize=7,facecolor=C['panel'],edgecolor=C['border'])
        [t.set_color(C['text']) for t in leg.get_texts()]
        ax5.grid(color=C['border'],alpha=0.5)

        ax6=self.fig.add_subplot(gs[1,2],projection='3d'); ax6.set_facecolor(C['panel'])
        X=np.linspace(-half*px,half*px,2*half); Xg,Yg=np.meshgrid(X,X)
        ax6.plot_surface(Xg,Yg,tip,cmap='plasma',alpha=0.9,linewidth=0)
        ax6.set_xlabel('X(nm)',color=C['muted'],fontsize=6,labelpad=1)
        ax6.set_ylabel('Y(nm)',color=C['muted'],fontsize=6,labelpad=1)
        ax6.set_zlabel('nm',   color=C['muted'],fontsize=6,labelpad=1)
        ax6.tick_params(colors=C['muted'],labelsize=5)
        ax6.set_title('Tip 3D',color=C['text'],fontsize=9,pad=3)

        r_str=f'{r_est:.0f}nm' if not np.isnan(r_est) else 'N/A'
        self.fig.suptitle(
            f'Blind Tip Reconstruction  |  {os.path.basename(m["filepath"])}  '
            f'|  {shape}  |  est. tip radius: {r_str}',
            color=C['text'],fontsize=10,fontweight='bold')
        self.canvas.draw()

    def keep_fig(self):
        """將目前圖表複製一份存入歷史，並在工具列新增縮圖按鈕"""
        import copy, io
        from PIL import Image, ImageTk

        # 用 bytes 複製目前 figure（避免共享 axes 物件）
        buf = io.BytesIO()
        self.fig.savefig(buf, format='png', dpi=60, facecolor=C['bg'],
                         bbox_inches='tight')
        buf.seek(0)

        # 生成標籤
        afm_name = os.path.basename(self.afm_path.get()) if self.meta else 'preview'
        label = f"{len(self.fig_history)+1}. {afm_name}"

        # 存成 PNG bytes（輕量，不存整個 Figure 物件）
        hires_buf = io.BytesIO()
        self.fig.savefig(hires_buf, format='png', dpi=150, facecolor=C['bg'],
                         bbox_inches='tight')
        self.fig_history.append({'label': label, 'png': hires_buf.getvalue(),
                                  'thumb_png': buf.getvalue()})
        self._refresh_hist()
        self.log(f'📷 已保留：{label}')

    def _refresh_hist(self):
        """重繪歷史縮圖按鈕列"""
        import io
        from PIL import Image, ImageTk
        for w in self.hist_frame.winfo_children():
            w.destroy()
        for i, item in enumerate(self.fig_history):
            # 縮圖
            img = Image.open(io.BytesIO(item['thumb_png'])).resize((80,52))
            tk_img = ImageTk.PhotoImage(img)
            frame = tk.Frame(self.hist_frame, bg=C['border'], padx=1, pady=1)
            frame.pack(side='left', padx=4)
            btn = tk.Button(frame, image=tk_img, text=f"{i+1}",
                            compound='top', bg=C['panel'], fg=C['text'],
                            relief='flat', cursor='hand2',
                            font=('Helvetica',7),
                            command=lambda idx=i: self._show_hist(idx))
            btn.image = tk_img   # 防止 GC
            btn.pack()
            # 右鍵選單（匯出/刪除）
            menu = tk.Menu(self, tearoff=0, bg=C['panel'], fg=C['text'],
                           activebackground=C['hover'])
            menu.add_command(label='💾 匯出為 PNG',
                             command=lambda idx=i: self._export_hist(idx))
            menu.add_command(label='🗑 刪除',
                             command=lambda idx=i: self._delete_hist(idx))
            btn.bind('<Button-3>', lambda e, m=menu: m.tk_popup(e.x_root, e.y_root))

    def _show_hist(self, idx):
        """點擊縮圖 → 在主畫布顯示該歷史圖片"""
        import io
        from PIL import Image
        import matplotlib.image as mpimg
        item = self.fig_history[idx]
        img = mpimg.imread(io.BytesIO(item['png']))
        self.fig.clear()
        ax = self.fig.add_axes([0,0,1,1])
        ax.imshow(img); ax.axis('off')
        self.fig.patch.set_facecolor(C['bg'])
        self.canvas.draw()
        self.log(f'顯示已保留圖片：{item["label"]}')

    def _export_hist(self, idx):
        """匯出歷史圖片為 PNG"""
        item = self.fig_history[idx]
        default_name = item['label'].replace('. ','_').replace(' ','_') + '.png'
        p = filedialog.asksaveasfilename(title='匯出圖片', defaultextension='.png',
            initialfile=default_name,
            filetypes=[('PNG','*.png'),('All','*.*')])
        if not p: return
        with open(p,'wb') as f: f.write(item['png'])
        self.log(f'✓ 匯出：{os.path.basename(p)}')

    def _delete_hist(self, idx):
        """刪除一筆歷史"""
        label = self.fig_history[idx]['label']
        self.fig_history.pop(idx)
        self._refresh_hist()
        self.log(f'🗑 刪除：{label}')

    def save(self):
        if self.tip is None:
            messagebox.showwarning('提示','請先執行重建'); return
        p=filedialog.asksaveasfilename(title='儲存 tip.mat',defaultextension='.mat',
            initialfile='tip_estimated.mat',filetypes=[('MATLAB','*.mat'),('All','*.*')])
        if not p: return
        try:
            savemat(p,{'tip':self.tip,'px_nm':self.meta['px_nm'],
                       'shape_type':self.shape_type,
                       'd_bot_nm':self.geom['d_bot'],'d_top_nm':self.geom['d_top'],
                       'height_nm':self.geom['height'],'pitch_nm':self.geom['pitch'],
                       'scan_size_nm':self.meta['scan_nm'],'n_px':self.meta['n_px'],
                       'patch_half_px':self.tip_half})
            png=p.replace('.mat','_report.png')
            self.fig.savefig(png,dpi=150,facecolor=C['bg'],bbox_inches='tight')
            self.log(f'✓ {os.path.basename(p)} + report PNG')
            messagebox.showinfo('完成',f'已儲存：\n{p}\n\n報告：\n{png}')
        except Exception as e:
            messagebox.showerror('失敗',str(e)); self.log(f'✗ {e}')

    def on_close(self):
        plt.close('all'); self.destroy()

if __name__ == '__main__':
    App().mainloop()