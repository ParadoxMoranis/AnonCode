import fitz
import re
import io
import os
import shutil
import subprocess
import tempfile
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict, Any, Optional
from PIL import Image

# [V10] LaTeX到Unicode数学符号映射
LATEX_TO_UNICODE = {
    r'\Gamma': 'Γ', r'\Delta': 'Δ', r'\Theta': 'Θ', r'\Lambda': 'Λ',
    r'\Xi': 'Ξ', r'\Pi': 'Π', r'\Sigma': 'Σ', r'\Phi': 'Φ',
    r'\Psi': 'Ψ', r'\Omega': 'Ω',
    r'\alpha': 'α', r'\beta': 'β', r'\gamma': 'γ', r'\delta': 'δ',
    r'\epsilon': 'ε', r'\theta': 'θ', r'\lambda': 'λ', r'\mu': 'μ',
    r'\nu': 'ν', r'\pi': 'π', r'\rho': 'ρ', r'\sigma': 'σ',
    r'\tau': 'τ', r'\phi': 'φ', r'\chi': 'χ', r'\psi': 'ψ', r'\omega': 'ω',
    r'\times': '×', r'\leq': '≤', r'\geq': '≥', r'\neq': '≠',
    r'\in': '∈', r'\subset': '⊂', r'\cup': '∪', r'\cap': '∩',
    r'\forall': '∀', r'\exists': '∃', r'\infty': '∞',
    r'\sum': '∑', r'\prod': '∏', r'\int': '∫',
    r'\coloneqq': '≔', r'\colon': ':', r'\to': '→', r'\rightarrow': '→',
    r'\mathbb{N}': 'ℕ', r'\mathbb{Z}': 'ℤ', r'\mathbb{Q}': 'ℚ',
    r'\mathbb{R}': 'ℝ', r'\mathbb{C}': 'ℂ',
    r'\mid': '|', r'\ldots': '…', r'\geqslant': '≥', r'\leqslant': '≤',
}

_LATEX_TOOLCHAIN_AVAILABLE = None


def latex_toolchain_available() -> bool:
    global _LATEX_TOOLCHAIN_AVAILABLE
    if _LATEX_TOOLCHAIN_AVAILABLE is None:
        _LATEX_TOOLCHAIN_AVAILABLE = all(shutil.which(cmd) for cmd in ('latex', 'dvipng'))
    return bool(_LATEX_TOOLCHAIN_AVAILABLE)


def normalize_latex_commands(latex_str: str) -> str:
    if not latex_str:
        return latex_str

    def _merge_split_command(match):
        parts = re.split(r'\s+', match.group(1).strip())
        if len(parts) < 2 or any(len(part) <= 1 for part in parts):
            return '\\' + match.group(1)
        return '\\' + ''.join(parts)

    result = latex_str
    result = re.sub(r'\\([A-Za-z]+(?:\s+[A-Za-z]+)+)', _merge_split_command, result)
    result = re.sub(r'\\([A-Za-z]+)\s+\{', r'\\\1{', result)
    result = re.sub(r'\\(begin|end)\s+\{', r'\\\1{', result)
    result = re.sub(r'\\math\s*bb', r'\\mathbb', result)
    result = re.sub(r'\\col\s*oneqq', r'\\coloneqq', result)
    result = re.sub(r'\\eq\s*colon', r'\\eqcolon', result)
    result = re.sub(r'\\leq\s*slant', r'\\leqslant', result)
    result = re.sub(r'\\geq\s*slant', r'\\geqslant', result)
    result = re.sub(r'\$\s+', '$', result)
    result = re.sub(r'\s+\$', '$', result)
    result = re.sub(r'\{\s+', '{', result)
    result = re.sub(r'\s+\}', '}', result)
    result = re.sub(r'\^\s+\{', '^{', result)
    result = re.sub(r'_\s+\{', '_{', result)
    standalone_commands = [
        r'\Gamma', r'\Delta', r'\Theta', r'\Lambda', r'\Xi', r'\Pi', r'\Sigma', r'\Phi', r'\Psi', r'\Omega',
        r'\alpha', r'\beta', r'\gamma', r'\delta', r'\epsilon', r'\theta', r'\lambda', r'\mu', r'\nu',
        r'\pi', r'\rho', r'\sigma', r'\tau', r'\phi', r'\chi', r'\psi', r'\omega',
        r'\times', r'\leq', r'\geq', r'\neq', r'\in', r'\subset', r'\cup', r'\cap', r'\forall', r'\exists',
        r'\infty', r'\sum', r'\prod', r'\int', r'\mid', r'\ldots', r'\geqslant', r'\leqslant',
        r'\to', r'\rightarrow', r'\leftarrow', r'\leftrightarrow', r'\backslash', r'\prime',
    ]
    for cmd in sorted(standalone_commands, key=len, reverse=True):
        result = re.sub(rf'({re.escape(cmd)})(?=[A-Za-z0-9])', rf'\1 ', result)
    result = simplify_inline_latex_structure(result)
    return result.strip()


def sanitize_inline_math_text(text: str) -> str:
    if not text:
        return text

    def repl(match):
        formula = match.group(0)
        inner = formula[1:-1]
        normalized = normalize_latex_commands(inner)
        return f"${normalized}$"

    return re.sub(r'(?<!\\)\$(?:\\.|[^$])+(?<!\\)\$', repl, text)


def strip_redundant_outer_braces(text: str) -> str:
    result = (text or "").strip()
    while result.startswith('{') and result.endswith('}'):
        depth = 0
        balanced = True
        for idx, ch in enumerate(result):
            if ch == '{':
                depth += 1
            elif ch == '}':
                depth -= 1
                if depth == 0 and idx != len(result) - 1:
                    balanced = False
                    break
        if not balanced or depth != 0:
            break
        result = result[1:-1].strip()
    return result


def simplify_inline_latex_structure(latex_str: str) -> str:
    if not latex_str:
        return latex_str

    result = latex_str
    env_patterns = (
        'array',
        'aligned',
        'alignedat',
        'split',
        'gathered',
        'smallmatrix',
    )

    for env in env_patterns:
        pattern = re.compile(
            rf'\\begin\{{{env}\}}\s*(?:\{{[^{{}}]*\}})?\s*(.*?)\s*\\end\{{{env}\}}',
            re.DOTALL,
        )

        def unwrap(match):
            inner = strip_redundant_outer_braces(match.group(1))
            # 单行、无对齐符时，这类环境通常只是 OCR 套出的冗余包装。
            if '\\\\' not in inner and '&' not in inner:
                return inner
            return match.group(0)

        result = pattern.sub(unwrap, result)

    return strip_redundant_outer_braces(result)


def crop_transparent_image(img: Image.Image) -> Tuple[Image.Image, Tuple[int, int, int, int]]:
    if img.mode != 'RGBA':
        img = img.convert('RGBA')
    alpha = img.getchannel('A')
    bbox = alpha.getbbox()
    if not bbox:
        return img, (0, 0, img.width, img.height)
    return img.crop(bbox), bbox


def _parse_tex_dimension_pt(log_text: str, key: str) -> Optional[float]:
    match = re.search(rf'{key}=([0-9.+-]+)pt', log_text)
    if not match:
        return None
    try:
        return float(match.group(1))
    except Exception:
        return None


def render_latex_to_image_via_system_latex_metadata(
    latex_str: str,
    font_size: float = 10.5,
    dpi: int = 300,
) -> Optional[Dict[str, Any]]:
    if not latex_toolchain_available():
        return None

    latex_body = normalize_latex_commands(latex_str).strip().strip('$').strip()
    latex_body = simplify_inline_latex_structure(latex_body)
    if not latex_body:
        return None

    baseline_skip = max(font_size * 1.2, font_size + 1.0)
    tex_source = rf"""\documentclass{{article}}
\pagestyle{{empty}}
\usepackage{{amsmath,amssymb,amsfonts}}
\begin{{document}}
\setbox0=\hbox{{{{\fontsize{{{font_size:.2f}}}{{{baseline_skip:.2f}}}\selectfont
${latex_body}$
}}}}
\typeout{{AGENTPARSE_WD=\the\wd0}}
\typeout{{AGENTPARSE_HT=\the\ht0}}
\typeout{{AGENTPARSE_DP=\the\dp0}}
\box0
\end{{document}}
"""
    try:
        with tempfile.TemporaryDirectory(prefix="agentparse_formula_") as tmpdir:
            tex_path = os.path.join(tmpdir, "formula.tex")
            dvi_path = os.path.join(tmpdir, "formula.dvi")
            png_path = os.path.join(tmpdir, "formula.png")
            with open(tex_path, "w", encoding="utf-8") as f:
                f.write(tex_source)

            env = os.environ.copy()
            env.setdefault("TEXMFOUTPUT", tmpdir)
            latex_proc = subprocess.run(
                ["latex", "-interaction=nonstopmode", "-halt-on-error", "formula.tex"],
                cwd=tmpdir,
                env=env,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            subprocess.run(
                ["dvipng", "-D", str(int(dpi)), "-T", "tight", "-bg", "Transparent", "-o", png_path, dvi_path],
                cwd=tmpdir,
                env=env,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            if not os.path.exists(png_path):
                return None

            raw_img = Image.open(png_path).convert("RGBA")
            img, crop_box = crop_transparent_image(raw_img)
            pt_per_px = 72.0 / float(dpi)
            width_pt = img.width * pt_per_px
            height_pt = img.height * pt_per_px

            log_text = (latex_proc.stdout or "") + "\n" + (latex_proc.stderr or "")
            width_metric = _parse_tex_dimension_pt(log_text, "AGENTPARSE_WD")
            ascent_metric = _parse_tex_dimension_pt(log_text, "AGENTPARSE_HT")
            depth_metric = _parse_tex_dimension_pt(log_text, "AGENTPARSE_DP")

            crop_top_pt = crop_box[1] * pt_per_px
            crop_bottom_pt = (raw_img.height - crop_box[3]) * pt_per_px
            if ascent_metric is not None:
                ascent_metric = max(0.0, ascent_metric - crop_top_pt)
            if depth_metric is not None:
                depth_metric = max(0.0, depth_metric - crop_bottom_pt)

            return {
                'img': img,
                'width_pt': float(width_metric if width_metric is not None else width_pt),
                'height_pt': float(height_pt),
                'ascent_pt': ascent_metric,
                'depth_pt': depth_metric,
                'formula': latex_body,
            }
    except Exception:
        return None


def render_latex_to_image_via_system_latex(latex_str: str, font_size: float = 10.5, dpi: int = 300) -> Tuple[Image.Image, float, float]:
    meta = render_latex_to_image_via_system_latex_metadata(latex_str, font_size=font_size, dpi=dpi)
    if not meta:
        return None, 0, 0
    return meta['img'], meta['width_pt'], meta['height_pt']

def preprocess_latex_for_matplotlib(latex_str: str) -> str:
    """
    [V22终极版] 预处理LaTeX，替换matplotlib不支持的命令为Unicode
    确保100%兼容性
    """
    import re
    
    # matplotlib不支持的符号映射（完整版）
    UNSUPPORTED_LATEX = {
        # 冒号变体（最常见的问题）
        r'\coloneqq': '≔',
        r'\eqqcolon': '≕',
        r'\coloneq': '≔',
        r'\eqcolon': '≕',
        r'\colonapprox': ':≈',
        r'\Coloneqq': '::=',
        r'\colon': ':',  # 最基础的
        
        # 比较符号（常见问题）
        r'\geqslant': '≥',
        r'\leqslant': '≤',
        r'\geqq': '≧',
        r'\leqq': '≦',
        r'\gneqq': '≩',
        r'\lneqq': '≨',
        r'\gtrless': '≷',
        r'\lessgtr': '≶',
        
        # 框和盒子符号
        r'\square': '□',
        r'\Box': '□',
        r'\blacksquare': '■',
        r'\boxed': '',
        
        # 集合符号
        r'\varnothing': '∅',
        r'\emptyset': '∅',
        r'\complement': 'ᶜ',
        
        # 箭头变体
        r'\longrightarrow': '→',
        r'\longleftarrow': '←',
        r'\longleftrightarrow': '↔',
        r'\Longrightarrow': '⇒',
        r'\Longleftarrow': '⇐',
        r'\Longleftrightarrow': '⇔',
        r'\mapsto': '↦',
        r'\longmapsto': '⟼',
        r'\hookrightarrow': '↪',
        r'\hookleftarrow': '↩',
        
        # 其他常见符号
        r'\mid': '|',
        r'\nmid': '∤',
        r'\parallel': '∥',
        r'\nparallel': '∦',
        r'\perp': '⊥',
        r'\angle': '∠',
    }
    
    result = normalize_latex_commands(latex_str)
    
    # 按长度降序排序，先替换长的（避免\coloneqq被\colon截断）
    sorted_items = sorted(UNSUPPORTED_LATEX.items(), key=lambda x: len(x[0]), reverse=True)
    
    for latex_cmd, unicode_char in sorted_items:
        result = result.replace(latex_cmd, unicode_char)
    
    # 处理\boxed{...}，移除boxed保留内容
    result = re.sub(r'\\boxed\{([^}]+)\}', r'\1', result)
    
    return result

def render_latex_to_image(latex_str: str, font_size: float = 10.5, dpi: int = 150) -> Tuple[Image.Image, float, float]:
    """
    [V22终极] 将LaTeX公式渲染为PNG图片，预处理+多层fallback
    
    Args:
        latex_str: LaTeX公式字符串（不含$符号）
        font_size: 字体大小（pt）
        dpi: 渲染DPI
    
    Returns:
        (PIL Image对象, 宽度pt, 高度pt)
    """
    try:
        system_meta = render_latex_to_image_via_system_latex_metadata(latex_str, font_size=font_size, dpi=max(dpi, 300))
        if system_meta:
            return system_meta['img'], system_meta['width_pt'], system_meta['height_pt']

        # [V22关键] 预处理：替换所有不支持的LaTeX命令
        processed_latex = preprocess_latex_for_matplotlib(latex_str)
        
        # 创建matplotlib figure
        fig = plt.figure(figsize=(0.1, 0.1))
        fig.patch.set_alpha(0)  # 透明背景
        
        # 渲染LaTeX
        text_obj = fig.text(0.5, 0.5, f'${processed_latex}$', 
                           ha='center', va='center',
                           fontsize=font_size,
                           usetex=False)  # 使用matplotlib内置的mathtext
        
        # 获取渲染后的边界框
        fig.canvas.draw()
        bbox = text_obj.get_window_extent(fig.canvas.get_renderer())
        
        # 调整figure大小以适配文本
        width_inch = bbox.width / dpi + 0.1
        height_inch = bbox.height / dpi + 0.1
        fig.set_size_inches(width_inch, height_inch)
        
        # 重新定位文本到中心
        text_obj.set_position((0.5, 0.5))
        
        # 保存到内存
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=dpi, bbox_inches='tight', 
                   pad_inches=0.05, transparent=True)
        buf.seek(0)
        
        # 转换为PIL Image并裁掉透明边
        img = Image.open(buf).convert("RGBA")
        img, _ = crop_transparent_image(img)
        
        # 计算实际尺寸（pt）
        width_pt = img.width * 72 / dpi
        height_pt = img.height * 72 / dpi
        
        plt.close(fig)
        
        return img, width_pt, height_pt
    
    except Exception as e:
        # [V15] 渲染失败时更详细的错误信息，但不打印（已预处理过）
        # print(f"    ⚠️ [LaTeX Render Failed] {latex_str[:50]}: {e}")
        return None, 0, 0

def convert_latex_to_unicode(text: str) -> str:
    """[V11修复] 将LaTeX公式转换为Unicode，正确处理花括号"""
    if not text:
        return text
    
    result = text
    
    # [关键修复] 先处理转义的花括号（必须最先）
    result = result.replace(r'\{', '{')
    result = result.replace(r'\}', '}')
    
    # 修复被空格拆分的LaTeX命令
    result = re.sub(r'\\col\s+oneqq', r'\\coloneqq', result)
    result = re.sub(r'\\col\s+on', r'\\colon', result)
    result = re.sub(r'\\math\s+bb', r'\\mathbb', result)
    
    # 替换LaTeX命令（按从长到短排序）
    for latex, unicode_char in sorted(LATEX_TO_UNICODE.items(), key=lambda x: len(x[0]), reverse=True):
        result = result.replace(latex, unicode_char)
    
    # 移除数学模式标记
    result = result.replace('$', '')
    
    # 清理LaTeX命令（保留花括号内容）
    result = re.sub(r'\\[a-zA-Z]+\{([^}]*)\}', r'\1', result)  # \command{text} -> text
    result = re.sub(r'\\[a-zA-Z]+', '', result)  # 移除其他命令
    
    # 清理多余空格
    result = re.sub(r'\s+', ' ', result)
    
    return result.strip()

class PDFReflowTool:
    def __init__(self, lang: str = 'zh'):
        self.DPI = 600
        self.lang = lang
        self.font_registry = {} 
        self.font_paths = {}
        self.min_font_size = 9.5   # [V18] 最小字号（避免过小）
        self.max_font_size = 11.5  # [V18] 最大字号（避免过大）
        self.default_font_size = 10.5  # [V18] 默认字号
        
        # [V12] LaTeX公式渲染缓存
        self.formula_cache = {}  # {latex_str: (pil_image, width, height)}
        
        # [WSL兼容] 获取脚本所在目录的绝对路径
        self._script_dir = os.path.dirname(os.path.abspath(__file__))
        self._base_dir = os.path.dirname(self._script_dir)  # agentParse 目录
        
        # [增强] 字体配置 - 支持 Regular/Bold/Italic/BoldItalic 四种变体
        # 格式: category -> variant -> [candidate paths]
        # 使用绝对路径，兼容 Windows 和 Linux/WSL
        fonts_dir = os.path.join(self._base_dir, 'fonts')
        
        self.font_config = {
            'zh': {
                'title': {
                    'regular':    [os.path.join(fonts_dir, 'SimHei.ttf'), os.path.join(fonts_dir, 'MSYHBD.TTF')],
                    'bold':       [os.path.join(fonts_dir, 'SimHei.ttf'), os.path.join(fonts_dir, 'MSYHBD.TTF')],  # 黑体本身已经较粗
                    'italic':     [os.path.join(fonts_dir, 'SimKai.ttf'), os.path.join(fonts_dir, '楷体_GB2312.ttf')],  # 中文用楷体代替斜体
                    'bolditalic': [os.path.join(fonts_dir, 'SimHei.ttf'), os.path.join(fonts_dir, 'MSYHBD.TTF')],
                },
                'heading': {
                    'regular':    [os.path.join(fonts_dir, 'SimHei.ttf'), os.path.join(fonts_dir, 'MSYH.TTF')],
                    'bold':       [os.path.join(fonts_dir, 'SimHei.ttf'), os.path.join(fonts_dir, 'MSYHBD.TTF')],
                    'italic':     [os.path.join(fonts_dir, 'SimKai.ttf'), os.path.join(fonts_dir, '楷体_GB2312.ttf')],
                    'bolditalic': [os.path.join(fonts_dir, 'SimHei.ttf'), os.path.join(fonts_dir, 'MSYHBD.TTF')],
                },
                'body': {
                    'regular':    [os.path.join(fonts_dir, 'SimSun.ttf'), os.path.join(fonts_dir, 'simsun.ttc')],
                    'bold':       [os.path.join(fonts_dir, 'SimHei.ttf'), os.path.join(fonts_dir, 'MSYHBD.TTF')],  # 宋体加粗用黑体
                    'italic':     [os.path.join(fonts_dir, 'SimKai.ttf'), os.path.join(fonts_dir, '楷体_GB2312.ttf')],  # 宋体斜体用楷体
                    'bolditalic': [os.path.join(fonts_dir, 'SimHei.ttf'), os.path.join(fonts_dir, 'MSYHBD.TTF')],
                },
                'caption': {
                    'regular':    [os.path.join(fonts_dir, 'SimKai.ttf'), os.path.join(fonts_dir, '楷体_GB2312.ttf')],
                    'bold':       [os.path.join(fonts_dir, 'SimHei.ttf'), os.path.join(fonts_dir, 'MSYHBD.TTF')],
                    'italic':     [os.path.join(fonts_dir, 'SimKai.ttf'), os.path.join(fonts_dir, '楷体_GB2312.ttf')],
                    'bolditalic': [os.path.join(fonts_dir, 'SimHei.ttf'), os.path.join(fonts_dir, 'MSYHBD.TTF')],
                },
                'fallback': 'china-s'
            },
            'en': {
                'title': {
                    'regular':    [os.path.join(fonts_dir, 'timesbd.ttf'), os.path.join(fonts_dir, 'Arial-Bold.ttf')],
                    'bold':       [os.path.join(fonts_dir, 'timesbd.ttf'), os.path.join(fonts_dir, 'Arial-Bold.ttf')],
                    'italic':     [os.path.join(fonts_dir, 'timesbi.ttf'), os.path.join(fonts_dir, 'Arial-BoldItalic.ttf')],
                    'bolditalic': [os.path.join(fonts_dir, 'timesbi.ttf'), os.path.join(fonts_dir, 'Arial-BoldItalic.ttf')],
                },
                'heading': {
                    'regular':    [os.path.join(fonts_dir, 'timesbd.ttf'), os.path.join(fonts_dir, 'Arial-Bold.ttf')],
                    'bold':       [os.path.join(fonts_dir, 'timesbd.ttf'), os.path.join(fonts_dir, 'Arial-Bold.ttf')],
                    'italic':     [os.path.join(fonts_dir, 'timesbi.ttf'), os.path.join(fonts_dir, 'Arial-BoldItalic.ttf')],
                    'bolditalic': [os.path.join(fonts_dir, 'timesbi.ttf'), os.path.join(fonts_dir, 'Arial-BoldItalic.ttf')],
                },
                'body': {
                    'regular':    [os.path.join(fonts_dir, 'times.ttf'), os.path.join(fonts_dir, 'Times.ttf')],
                    'bold':       [os.path.join(fonts_dir, 'timesbd.ttf'), os.path.join(fonts_dir, 'Times-Bold.ttf')],
                    'italic':     [os.path.join(fonts_dir, 'timesi.ttf'), os.path.join(fonts_dir, 'Times-Italic.ttf')],
                    'bolditalic': [os.path.join(fonts_dir, 'timesbi.ttf'), os.path.join(fonts_dir, 'Times-BoldItalic.ttf')],
                },
                'caption': {
                    'regular':    [os.path.join(fonts_dir, 'timesi.ttf'), os.path.join(fonts_dir, 'Times-Italic.ttf')],
                    'bold':       [os.path.join(fonts_dir, 'timesbi.ttf'), os.path.join(fonts_dir, 'Times-BoldItalic.ttf')],
                    'italic':     [os.path.join(fonts_dir, 'timesi.ttf'), os.path.join(fonts_dir, 'Times-Italic.ttf')],
                    'bolditalic': [os.path.join(fonts_dir, 'timesbi.ttf'), os.path.join(fonts_dir, 'Times-BoldItalic.ttf')],
                },
                'fallback': 'tiro'
            }
        }
        
        # [增强] 加载所有字体变体
        target_fonts = self.font_config.get(lang, self.font_config['en'])
        for cat, variants in target_fonts.items():
            if cat == 'fallback': 
                continue
            for variant, paths in variants.items():
                key = f"{cat}_{variant}"  # e.g., "body_bold", "title_italic"
                for p in paths:
                    if os.path.exists(p):
                        try:
                            with open(p, "rb") as f:
                                self.font_registry[key] = f.read()
                            self.font_paths[key] = p
                            break
                        except: 
                            continue
                # 兼容旧的 key (没有变体后缀)
                if variant == 'regular' and key in self.font_registry:
                    self.font_registry[cat] = self.font_registry[key]
                    self.font_paths[cat] = self.font_paths[key]
        
        # 加载计算用字体 (优先用 Body 字体以保证宽度计算准确)
        if 'body_regular' in self.font_registry:
            self.calc_font = fitz.Font(fontbuffer=self.font_registry['body_regular'])
        elif 'body' in self.font_registry:
            self.calc_font = fitz.Font(fontbuffer=self.font_registry['body'])
        else:
            self.calc_font = fitz.Font(target_fonts['fallback'])
            
        self.latex_family = "serif" if lang == 'zh' else "sans-serif"
        
        print(f"[PDFReflowTool] Loaded {len(self.font_registry)} font variants for '{lang}'")

    def _get_font_for_style(self, font_key: str, is_bold: bool = False, is_italic: bool = False) -> Tuple[bytes, str, str]:
        """
        [新增] 根据样式获取正确的字体
        返回: (font_buffer, font_path, font_name)
        """
        # 确定变体名
        if is_bold and is_italic:
            variant = 'bolditalic'
        elif is_bold:
            variant = 'bold'
        elif is_italic:
            variant = 'italic'
        else:
            variant = 'regular'
        
        # 尝试获取对应变体
        key_with_variant = f"{font_key}_{variant}"
        
        if key_with_variant in self.font_registry:
            return (
                self.font_registry[key_with_variant],
                self.font_paths.get(key_with_variant),
                key_with_variant
            )
        
        # 回退到 regular 变体
        key_regular = f"{font_key}_regular"
        if key_regular in self.font_registry:
            return (
                self.font_registry[key_regular],
                self.font_paths.get(key_regular),
                key_regular
            )
        
        # 再回退到无后缀的 key (兼容旧格式)
        if font_key in self.font_registry:
            return (
                self.font_registry[font_key],
                self.font_paths.get(font_key),
                font_key
            )
        
        # 最终回退到 fallback
        fallback = self.font_config[self.lang].get('fallback', 'china-s')
        return (None, None, fallback)

    def _normalize_rgb_color(self, color) -> Optional[Tuple[float, float, float]]:
        if not isinstance(color, (tuple, list)) or len(color) < 3:
            return None
        values = []
        for channel in color[:3]:
            try:
                channel = float(channel)
            except Exception:
                return None
            if channel > 1.0:
                channel /= 255.0
            values.append(max(0.0, min(1.0, channel)))
        return tuple(values)

    def _is_near_white(self, color: Optional[Tuple[float, float, float]], threshold: float = 0.97) -> bool:
        if not color:
            return False
        return all(channel >= threshold for channel in color)

    def _is_near_black(self, color: Optional[Tuple[float, float, float]], threshold: float = 0.16) -> bool:
        if not color:
            return False
        return all(channel <= threshold for channel in color)

    def _expand_rect_to_page(self, page, rect: fitz.Rect, margin: float = 0.0) -> fitz.Rect:
        page_rect = page.rect
        return fitz.Rect(
            max(page_rect.x0, rect.x0 - margin),
            max(page_rect.y0, rect.y0 - margin),
            min(page_rect.x1, rect.x1 + margin),
            min(page_rect.y1, rect.y1 + margin),
        )

    def _color_distance(
        self,
        left: Optional[Tuple[float, float, float]],
        right: Optional[Tuple[float, float, float]],
    ) -> float:
        if not left or not right:
            return 1.0
        return sum(abs(float(a) - float(b)) for a, b in zip(left[:3], right[:3])) / 3.0

    def _score_vector_fill_candidate(
        self,
        page,
        rect: fitz.Rect,
        candidate_rect: fitz.Rect,
        overlap_ratio: float,
    ) -> Tuple[Tuple[int, float, float, float], bool]:
        expanded_rect = self._expand_rect_to_page(page, rect, margin=0.2)
        contains_rect = candidate_rect.contains(expanded_rect) or overlap_ratio >= 0.995
        rect_area = max(1.0, rect.get_area())
        candidate_area = max(candidate_rect.get_area(), 1.0)
        extra_area = max(0.0, (candidate_area / rect_area) - 1.0)
        specificity = 1.0 / (1.0 + extra_area)
        return (
            (
                1 if contains_rect else 0,
                round(overlap_ratio, 6),
                round(specificity, 6),
                -round(extra_area, 6),
            ),
            contains_rect,
        )

    def _sample_background_color(self, page, bbox, sample_margin: float = 1.5) -> Optional[Tuple[float, float, float]]:
        try:
            rect = fitz.Rect(bbox)
            sample_rect = self._expand_rect_to_page(page, rect, sample_margin)
            pix = page.get_pixmap(clip=sample_rect, dpi=72, alpha=False)
            if pix.width < 3 or pix.height < 3:
                return None

            mode = "RGB" if pix.n < 4 else "RGBA"
            img = Image.frombytes(mode, [pix.width, pix.height], pix.samples)
            if img.mode != "RGB":
                img = img.convert("RGB")

            border = max(1, min(3, min(img.size) // 8))
            width, height = img.size
            counts = {}
            for y in range(height):
                for x in range(width):
                    if border <= x < width - border and border <= y < height - border:
                        continue
                    rgb = img.getpixel((x, y))[:3]
                    quantized = tuple(int(round(channel / 8.0) * 8) for channel in rgb)
                    counts[quantized] = counts.get(quantized, 0) + 1

            if not counts:
                return None

            dominant = max(counts.items(), key=lambda item: item[1])[0]
            normalized = tuple(max(0.0, min(1.0, channel / 255.0)) for channel in dominant)
            if self._is_near_white(normalized):
                return None
            return normalized
        except Exception:
            return None

    def _find_enclosing_vector_fill(
        self,
        page,
        bbox,
        min_overlap_ratio: float = 0.72,
    ) -> Optional[Tuple[float, float, float]]:
        try:
            rect = fitz.Rect(bbox)
            rect_area = max(1.0, rect.get_area())
            best_fill = None
            best_score = None

            for drawing in page.get_drawings():
                fill = drawing.get("fill")
                draw_rect = drawing.get("rect")
                if not fill or not draw_rect:
                    continue
                candidate_rect = fitz.Rect(draw_rect)
                if not candidate_rect.intersects(rect):
                    continue
                intersection = candidate_rect & rect
                intersection_area = intersection.get_area() if intersection else 0.0
                if intersection_area <= 0:
                    continue

                overlap_ratio = intersection_area / rect_area
                if overlap_ratio < min_overlap_ratio:
                    continue

                normalized_fill = tuple(float(c) for c in fill[:3])
                if self._is_near_white(normalized_fill):
                    continue

                candidate_area = max(candidate_rect.get_area(), 1.0)
                area_ratio = candidate_area / rect_area
                # 只接受与文本框贴合的底色，避免把大图/整页色块当作文本背景。
                if area_ratio > 2.6:
                    continue
                if overlap_ratio < 0.86 and area_ratio > 1.6:
                    continue
                score, contains_rect = self._score_vector_fill_candidate(page, rect, candidate_rect, overlap_ratio)

                # 只把真正包裹文本框的色块当作背景，避免把黑色字形/细线误判成底色后整块盖黑。
                if not contains_rect and candidate_area < rect_area * 1.35:
                    continue
                if self._is_near_black(normalized_fill) and not contains_rect:
                    continue

                if best_fill is None or best_score is None or score > best_score:
                    best_fill = normalized_fill
                    best_score = score

            return best_fill
        except Exception:
            return None

    def _resolve_cleanup_fill_color(self, page, bbox, bg_color=None) -> Optional[Tuple[float, float, float]]:
        explicit = self._normalize_rgb_color(bg_color)
        vector_fill = self._find_enclosing_vector_fill(page, bbox)
        if vector_fill is not None:
            if explicit and not self._is_near_white(explicit):
                if self._color_distance(explicit, vector_fill) <= 0.035:
                    return explicit
            return vector_fill
        # 旧缓存里的显式 bg_color 也可能来自过宽松的背景检测。
        # 没有当前页矢量底色证据时，宁可丢弃背景样式，也不冒险盖住正文。
        return None

    def _paint_over_text_area(self, page, bbox, fill_color, margin: float = 0.22):
        rect = fitz.Rect(bbox)
        cover_rect = self._expand_rect_to_page(page, rect, margin=margin)
        try:
            page.draw_rect(
                cover_rect,
                color=fill_color,
                fill=fill_color,
                width=0,
                overlay=True,
                stroke_opacity=0,
                fill_opacity=1,
            )
        except Exception:
            pass

    def _deep_clean_area_with_bg(self, page, bbox, bg_color=None):
        """
        [增强] 清理区域并可选地填充背景色
        """
        rect = fitz.Rect(bbox)
        fill_color = self._resolve_cleanup_fill_color(page, rect, bg_color)

        # 对于有明确原始色块承载的文本，只覆盖文字区域，保留外层卡片/圆角/描边
        if fill_color is not None:
            self._paint_over_text_area(page, rect, fill_color, margin=0.22)
            return

        clean_rect = self._expand_rect_to_page(page, rect, margin=0.35)
        fill_color = (1, 1, 1)

        page.add_redact_annot(clean_rect, fill=fill_color)
        page.apply_redactions(
            images=fitz.PDF_REDACT_IMAGE_NONE,
            graphics=fitz.PDF_REDACT_LINE_ART_REMOVE_IF_TOUCHED,
            text=fitz.PDF_REDACT_TEXT_REMOVE,
        )
        try:
            page.draw_rect(
                clean_rect,
                color=fill_color,
                fill=fill_color,
                width=0,
                overlay=True,
                stroke_opacity=0,
                fill_opacity=1,
            )
        except Exception:
            pass

    # 检测系统是否有 LaTeX
    _latex_available = None
    
    @classmethod
    def _check_latex_available(cls):
        """检测系统是否安装了 LaTeX"""
        if cls._latex_available is None:
            cls._latex_available = latex_toolchain_available()
        return cls._latex_available

    def _is_font_size_locked(self, style: Dict) -> bool:
        return bool(style.get('_lock_font_size') or style.get('lock_font_size'))

    def _resolve_font_size(self, style: Dict, key: str = 'size', default: float = 10.5) -> float:
        font_size = float(style.get(key, default))
        if self._is_font_size_locked(style):
            return font_size
        if font_size < 8:
            return self.min_font_size
        if font_size > 14:
            return self.max_font_size
        return max(self.min_font_size, min(self.max_font_size, font_size))

    def _render_latex_to_pixmap(self, latex_str, fontsize):
        """
        渲染 LaTeX 公式为 Pixmap
        
        优先使用系统 LaTeX（效果更好）
        如果系统没有 LaTeX，使用 matplotlib 的 mathtext（无需外部依赖）
        """
        try:
            if not latex_str:
                return None, 0.0

            content_pure = re.sub(r'\\(label|ref|cite)\{.*?\}', '', latex_str.strip()).replace('$', '').strip()
            if not content_pure:
                return None, 0.0

            img, _, _ = render_latex_to_image(content_pure, font_size=fontsize, dpi=max(int(self.DPI * (fontsize / 10.0)), 300))
            if not img:
                return None, 0.0

            buf = io.BytesIO()
            img.save(buf, format='PNG')
            buf.seek(0)
            pix = fitz.Pixmap(buf)
            
            # 计算垂直偏移
            has_descent = any(x in content_pure for x in ['_', 'j', 'p', 'q', 'y', 'g', '\\rho', '\\mu', '\\chi'])
            has_ascent = any(x in content_pure for x in ['^', 't', 'h', 'l', 'k', 'b', '\\partial', '\\int'])
            offset_y_ratio = 0.12 if (has_descent and not has_ascent) else (-0.12 if (has_ascent and not has_descent) else 0.0)
            
            return pix, offset_y_ratio
        except Exception as e:
            plt.close('all')
            return None, 0.0

    def _balance_braces(self, latex_str: str) -> str:
        text = latex_str
        open_count = text.count('{')
        close_count = text.count('}')
        if open_count > close_count:
            text = text + ('}' * (open_count - close_count))
        return text

    def _build_formula_candidates(self, latex_content: str) -> List[str]:
        normalized = normalize_latex_commands(latex_content.strip().strip('$').strip())
        balanced = self._balance_braces(normalized)
        compact = re.sub(r'\s+', ' ', balanced).strip()
        no_space_after_cmd = re.sub(r'\\\s+', r'\\', compact)
        candidates = []
        simplified = simplify_inline_latex_structure(no_space_after_cmd)
        for candidate in (normalized, balanced, compact, no_space_after_cmd, simplified):
            if candidate and candidate not in candidates:
                candidates.append(candidate)
        return candidates

    def _compute_formula_offset_ratio(self, content_pure: str) -> float:
        has_descent = any(x in content_pure for x in ['_', 'j', 'p', 'q', 'y', 'g', '\\rho', '\\mu', '\\chi'])
        has_ascent = any(x in content_pure for x in ['^', 't', 'h', 'l', 'k', 'b', '\\partial', '\\int'])
        if has_descent and not has_ascent:
            return 0.10
        if has_ascent and not has_descent:
            return -0.08
        return 0.0

    def _resolve_inline_formula_dimensions(
        self,
        asset: Dict[str, Any],
        line_height: float,
        font_size: Optional[float] = None,
        max_width: Optional[float] = None,
    ) -> Tuple[float, float, float]:
        target_w = float(asset.get('width_pt', 0.0) or 0.0)
        target_h = float(asset.get('height_pt', 0.0) or 0.0)
        if target_w <= 0 or target_h <= 0:
            return target_w, target_h, 1.0

        scale = 1.0
        max_h = max(line_height * 0.86, 1.0)
        if font_size:
            max_h = min(max_h, max(font_size * 1.12, font_size * 0.82))
            formula_text = asset.get('formula', '').strip()
            if formula_text.startswith('^') or formula_text.startswith('_'):
                max_h = min(max_h, max(font_size * 0.72, 1.0))
        if target_h > max_h:
            scale = min(scale, max_h / target_h)
        if max_width and max_width > 0 and target_w > max_width:
            scale = min(scale, max_width / target_w)

        if scale < 1.0:
            target_w *= scale
            target_h *= scale
        return target_w, target_h, scale

    def _estimate_inline_formula_box(
        self,
        latex_content: str,
        font_size: float,
        line_height: float,
    ) -> Tuple[float, float]:
        normalized = normalize_latex_commands(latex_content or "")
        compact = re.sub(r"\s+", "", normalized)
        if not compact:
            return max(font_size * 0.8, 1.0), max(font_size * 1.05, 1.0)

        visible = re.sub(r"\\[A-Za-z]+", "m", compact)
        visible = re.sub(r"[{}]", "", visible)
        visible_len = max(1, len(visible))

        frac_like = any(cmd in normalized for cmd in ("\\frac", "\\dfrac", "\\tfrac", "\\binom"))
        tall_like = frac_like or any(
            cmd in normalized
            for cmd in ("\\sum", "\\prod", "\\int", "\\iint", "\\iiint", "\\sqrt", "\\overset", "\\underset")
        )
        script_count = normalized.count("^") + normalized.count("_")
        command_count = len(re.findall(r"\\[A-Za-z]+", normalized))

        width = visible_len * font_size * 0.46
        width += command_count * font_size * 0.14
        width += min(script_count, 4) * font_size * 0.12
        if frac_like:
            width *= 1.18

        height = font_size * 1.04
        if script_count:
            height = max(height, font_size * (1.12 + min(script_count, 3) * 0.08))
        if tall_like:
            height = max(height, font_size * 1.42)

        height = max(height, line_height * 0.92)
        return max(width, font_size * 0.8), max(height, font_size * 1.0)

    def _render_formula_asset(self, latex_content: str, font_size: float) -> Dict[str, Any]:
        cache_key = f"{latex_content}_{font_size}"
        cached = self.formula_cache.get(cache_key)
        if isinstance(cached, dict):
            return cached

        for candidate in self._build_formula_candidates(latex_content):
            meta = render_latex_to_image_via_system_latex_metadata(candidate, font_size=font_size, dpi=300)
            if not meta:
                img, img_w_pt, img_h_pt = render_latex_to_image(candidate, font_size, dpi=300)
                if img:
                    meta = {
                        'img': img,
                        'width_pt': img_w_pt,
                        'height_pt': img_h_pt,
                        'ascent_pt': None,
                        'depth_pt': None,
                    }
            if meta and meta.get('img'):
                img_bytes = io.BytesIO()
                meta['img'].save(img_bytes, format='PNG')
                img_bytes.seek(0)
                asset = {
                    'status': 'image',
                    'pix': fitz.Pixmap(img_bytes),
                    'width_pt': float(meta.get('width_pt', 0.0) or 0.0),
                    'height_pt': float(meta.get('height_pt', 0.0) or 0.0),
                    'ascent_pt': meta.get('ascent_pt'),
                    'depth_pt': meta.get('depth_pt'),
                    'offset_ratio': self._compute_formula_offset_ratio(candidate),
                    'formula': candidate,
                }
                self.formula_cache[cache_key] = asset
                return asset

        fallback_formula = self._build_formula_candidates(latex_content)[0] if latex_content else ""
        asset = {
            'status': 'fallback',
            'text': convert_latex_to_unicode(f"${fallback_formula}$") if fallback_formula else latex_content,
            'formula': fallback_formula,
        }
        self.formula_cache[cache_key] = asset
        return asset

    def _draw_formula_segment(
        self,
        page,
        rect: fitz.Rect,
        cursor_x: float,
        current_baseline_y: float,
        line_height: float,
        font_size: float,
        char_sp: float,
        latex_content: str,
        text_color=(0, 0, 0),
        line_count: int = 0,
        bottom_limit: Optional[float] = None,
    ) -> Tuple[float, float, int, bool]:
        asset = self._render_formula_asset(latex_content, font_size)
        render_bottom = bottom_limit if bottom_limit is not None else rect.y1 + max(1.5, min(4.0, font_size * 0.28))

        if asset.get('status') == 'image':
            target_w, target_h, scale = self._resolve_inline_formula_dimensions(
                asset,
                line_height,
                font_size=font_size,
                max_width=max(rect.width * 0.98, 1.0),
            )

            if cursor_x + target_w > rect.x1:
                cursor_x = rect.x0
                current_baseline_y += line_height
                line_count += 1
                if current_baseline_y > render_bottom:
                    return cursor_x, current_baseline_y, line_count, False

            depth_pt = asset.get('depth_pt')
            ascent_pt = asset.get('ascent_pt')
            if depth_pt is not None:
                depth_pt = float(depth_pt) * scale
            if ascent_pt is not None:
                ascent_pt = float(ascent_pt) * scale

            if depth_pt is not None:
                img_top = current_baseline_y - (target_h - depth_pt)
            elif ascent_pt is not None:
                img_top = current_baseline_y - ascent_pt
            else:
                text_visual_center_y = current_baseline_y - (font_size * 0.32)
                img_top = text_visual_center_y - (target_h / 2) + (target_h * asset.get('offset_ratio', 0.0))
            img_rect = fitz.Rect(cursor_x, img_top, cursor_x + target_w, img_top + target_h)
            if img_rect.y1 > render_bottom:
                return cursor_x, current_baseline_y, line_count, False
            page.insert_image(img_rect, pixmap=asset['pix'])
            cursor_x += target_w + char_sp + 1
            return cursor_x, current_baseline_y, line_count, True

        fallback_text = asset.get('text', '') or convert_latex_to_unicode(f"${latex_content}$")
        for char in fallback_text:
            w = self.calc_font.text_length(char, fontsize=font_size) if char.strip() else font_size * 0.3
            if cursor_x + w > rect.x1:
                cursor_x = rect.x0
                current_baseline_y += line_height
                line_count += 1
                if current_baseline_y > render_bottom:
                    return cursor_x, current_baseline_y, line_count, False
            try:
                page.insert_text((cursor_x, current_baseline_y), char, fontsize=font_size, color=text_color)
            except Exception:
                pass
            cursor_x += w + char_sp
        return cursor_x, current_baseline_y, line_count, True

    def _resolve_render_bottom_limit(self, rect: fitz.Rect, style: Dict, font_size: float) -> float:
        explicit = style.get('_render_bottom_limit')
        if explicit is None:
            explicit = rect.y1 + max(1.5, min(4.0, font_size * 0.28))
        return max(rect.y1, float(explicit))

    def _parse_segments(self, text: str) -> List[Tuple[str, str]]:
        # ... (保持不变) ...
        segments = []
        last = 0
        pattern = r'(?<!\\)\$(?:\\.|[^$])+(?<!\\)\$'
        for m in re.finditer(pattern, text):
            if m.start() > last: segments.append(('text', text[last:m.start()]))
            segments.append(('math', m.group(0)))
            last = m.end()
        if last < len(text): segments.append(('text', text[last:]))
        return segments

    def simulate_layout_metrics(self, bbox: List[float], text: str, style: Dict) -> Dict[str, Any]:
        """计算文本在BBox中的布局指标"""
        rect = fitz.Rect(bbox)
        width, height = rect.width, rect.height
        if width <= 0 or height <= 0:
            return {"status": "error", "fill_ratio": 0, "is_overflow": False}
        font_size = self._resolve_font_size(style)
        line_spacing = style.get('line', 1.35)
        char_spacing = style.get('char', 0.0)
        line_h = font_size * line_spacing

        cursor_x = 0.0
        current_line_visual_h = font_size * 1.06
        line_visual_heights = []
        line_count = 1 if text else 0
        segments = self._parse_segments(text)

        def flush_line():
            nonlocal cursor_x, current_line_visual_h, line_count
            line_visual_heights.append(max(line_h, current_line_visual_h))
            cursor_x = 0.0
            current_line_visual_h = font_size * 1.06
            line_count += 1

        for seg_type, content in segments:
            if seg_type == 'text':
                for char in content:
                    if char == '\n':
                        flush_line()
                        continue
                    w = self.calc_font.text_length(char, fontsize=font_size) if char.strip() else font_size * 0.3
                    visual_h = font_size * (1.10 if re.search(r'[\u4e00-\u9fff]', char) else 1.02)
                    if cursor_x > 0 and cursor_x + w > width:
                        flush_line()
                    cursor_x += w + char_spacing
                    current_line_visual_h = max(current_line_visual_h, visual_h)
            elif seg_type == 'math':
                math_w, math_h = self._estimate_inline_formula_box(
                    content.strip('$').strip(),
                    font_size=font_size,
                    line_height=line_h,
                )
                math_w = min(math_w, max(width * 0.98, 1.0))
                if cursor_x > 0 and cursor_x + math_w > width:
                    flush_line()
                cursor_x += math_w + char_spacing
                current_line_visual_h = max(current_line_visual_h, math_h * 1.04)

        if text:
            line_visual_heights.append(max(line_h, current_line_visual_h))

        total_h = sum(line_visual_heights) if line_visual_heights else 0.0
        return {
            "needed_height": total_h,
            "box_height": height,
            "fill_ratio": (total_h / height) if height > 0 else 0.0,
            "is_overflow": total_h > height,
            "line_count": len(line_visual_heights),
        }

    def _calculate_text_width(self, text: str, font_size: float, char_spacing: float = 0.0) -> float:
        """
        [锦囊一辅助] 计算文本的渲染宽度
        """
        total_width = 0.0
        segments = self._parse_segments(text)
        
        for seg_type, content in segments:
            if seg_type == 'text':
                content = content.replace('\n', '')
                for char in content:
                    total_width += self.calc_font.text_length(char, fontsize=font_size) + char_spacing
            elif seg_type == 'math':
                math_w, _ = self._estimate_inline_formula_box(
                    content.strip('$').strip(),
                    font_size=font_size,
                    line_height=font_size * 1.35,
                )
                total_width += math_w + char_spacing

        return total_width
    
    def _accordion_fit_style(self, bbox: List[float], text: str, style: Dict) -> Dict:
        """
        [V7 重构：通用自适应缩放] 使用真实布局模拟，确保文本100%能放下
        
        核心原则：
        1. 使用 simulate_layout_metrics 计算真实高度（用真实字体）
        2. 如果溢出，迭代缩小字号直到不溢出
        3. 最小字号限制：原字号的60%（保证可读性）
        4. 永不截断文本
        
        Args:
            bbox: 边界框 [x0, y0, x1, y1]
            text: 要渲染的文本
            style: 原始样式字典
            
        Returns:
            调整后的样式字典
        """
        adjusted_style = style.copy()
        if self._is_font_size_locked(style):
            adjusted_style['_accordion_action'] = 'locked'
            return adjusted_style
        
        rect = fitz.Rect(bbox)
        if rect.width <= 0 or rect.height <= 0:
            return adjusted_style
        
        font_size = self._resolve_font_size(style)
        # [V18] 最小字号不低于系统限制
        min_font_size = self.min_font_size
        
        # === 步骤1：检查当前样式是否溢出 ===
        metrics = self.simulate_layout_metrics(bbox, text, adjusted_style)
        
        if not metrics.get('is_overflow', False):
            # 不溢出，保持原样
            adjusted_style['_accordion_action'] = 'none'
            return adjusted_style
        
        # === 步骤2：迭代缩小字号直到不溢出 ===
        current_size = font_size
        best_style = adjusted_style.copy()
        
        for _ in range(20):  # 最多20次迭代
            # 计算需要的缩放比例
            needed_h = metrics.get('needed_height', rect.height)
            box_h = metrics.get('box_height', rect.height)
            
            if needed_h <= 0:
                break
                
            # 按比例缩小（稍微多缩一点，确保能放下）
            scale = (box_h / needed_h) * 0.95
            new_size = current_size * scale
            
            # 限制最小字号
            if new_size < min_font_size:
                new_size = min_font_size
            
            # 更新样式
            best_style = adjusted_style.copy()
            best_style['size'] = new_size
            # 行高按比例缩小
            best_style['line'] = style.get('line', 1.35)
            
            # 重新计算
            metrics = self.simulate_layout_metrics(bbox, text, best_style)
            current_size = new_size
            
            if not metrics.get('is_overflow', False):
                # 成功适配
                best_style['_accordion_action'] = 'scale_down'
                best_style['_accordion_scale'] = new_size / font_size
                return best_style
            
            # 已到最小字号，停止
            if new_size <= min_font_size:
                break
        
        # === 步骤3：如果缩到最小仍溢出，接受溢出但不截断 ===
        best_style['_accordion_action'] = 'overflow_accepted'
        best_style['_overflow_ratio'] = metrics.get('fill_ratio', 1.0)
        return best_style
    
    def _smart_truncate(self, text: str, bbox: List[float], style: Dict, max_iterations: int = 3) -> str:
        """
        [V4 修复] 极度保守的截断策略 - 几乎不截断正文
        
        核心原则：
        1. 只有严重溢出（>20%）才考虑截断
        2. 宁可让文本溢出 bbox，也不要丢失内容
        """
        rect = fitz.Rect(bbox)
        box_height = rect.height
        
        # 先用手风琴算法调整
        adjusted = self._accordion_fit_style(bbox, text, style)
        
        # 模拟布局
        metrics = self.simulate_layout_metrics(bbox, text, adjusted)
        fill_ratio = metrics.get('fill_ratio', 1.0)
        
        # [V4 修复] 只有溢出超过 20% 才截断，否则接受溢出
        if fill_ratio <= 1.20:
            return text  # 20% 以内的溢出可接受，不截断
        
        # 严重溢出（>20%）才截断，但只截断到 120%
        current_text = text
        ellipsis = "..."
        
        for _ in range(max_iterations):
            # 每次只截掉 5%（更保守）
            cut_len = max(1, int(len(current_text) * 0.05))
            current_text = current_text[:-cut_len].rstrip() + ellipsis
            
            metrics = self.simulate_layout_metrics(bbox, current_text, adjusted)
            # 截断到 120% 即可停止
            if metrics.get('fill_ratio', 1.0) <= 1.20:
                break
        
        return current_text

    def _deep_clean_area(self, page, bbox):
        self._deep_clean_area_with_bg(page, bbox, None)

    def draw_content(self, page, bbox, text, style, auto_fit: bool = True):
        """
        [增强版绘图 V2.1 - 手风琴算法]
        1. 支持 font_key 切换 (SimHei, SimKai 等)
        2. 支持粗体/斜体样式
        3. 支持文字颜色
        4. 支持背景色保留
        5. 修复单行标题因 BBox 过窄而不渲染的问题 (宽松裁剪)
        6. [新增] 自动调整字号/字间距适配 BBox (手风琴算法)
        7. [V6 修复] 标题类型不截断，宁可缩小字号
        
        Args:
            page: PDF 页面对象
            bbox: 边界框
            text: 要渲染的文本
            style: 样式字典
            auto_fit: 是否启用自动适配 (手风琴算法)
        """
        # [锦囊一] 自动适配样式（但不截断文本）
        if auto_fit:
            style = self._accordion_fit_style(bbox, text, style)
            
            # [V5 修复] 完全禁用文本截断，宁可溢出也不丢失内容
            # 截断会导致翻译内容丢失，破坏文档完整性
            # text = self._smart_truncate(text, bbox, style)  # 已禁用
        
        # [V17] 保持LaTeX原样，交给图片渲染
        # text = convert_latex_to_unicode(text)  # 禁用
        
        # 提取样式信息 + [V18] 字号归一化
        font_size = self._resolve_font_size(style)
        line_height = font_size * style.get('line', 1.35)
        char_sp = style.get('char', 0.0)
        font_key = style.get('font_key', 'body')
        
        # [新增] 提取粗体/斜体/颜色/背景色
        is_bold = style.get('is_bold', False)
        is_italic = style.get('is_italic', False)
        text_color = style.get('text_color', (0, 0, 0))
        bg_color = style.get('bg_color', None)
        
        # 确保颜色格式正确 (0-1 范围的 RGB 元组)
        if text_color and isinstance(text_color, (tuple, list)) and len(text_color) >= 3:
            text_color = tuple(float(c) for c in text_color[:3])
            # 如果颜色值大于1，假设是 0-255 范围，需要转换
            if any(c > 1.0 for c in text_color):
                text_color = tuple(c / 255.0 for c in text_color)
        else:
            text_color = (0, 0, 0)
        
        # [新增] 使用带背景色的清理方法
        self._deep_clean_area_with_bg(page, bbox, bg_color)
        
        rect = fitz.Rect(bbox)
        x0, y0 = rect.tl
        
        # [新增] 根据样式获取正确的字体变体
        font_buffer, font_file, font_name = self._get_font_for_style(font_key, is_bold, is_italic)

        ascent = font_size * 0.88
        bottom_limit = self._resolve_render_bottom_limit(rect, style, font_size)
        
        # [关键修复] 垂直对齐策略
        # 如果是单行文本且高度不足，强制垂直居中，防止基线溢出
        is_likely_single_line = len(text) * font_size < rect.width * 1.5
        
        start_y_offset = (line_height - font_size) / 2
        
        if is_likely_single_line and rect.height < line_height:
            # 盒子太矮，强制居中对齐
            current_baseline_y = y0 + (rect.height - font_size) / 2 + ascent
        else:
            # 标准顶对齐
            current_baseline_y = y0 + ascent + start_y_offset

        cursor_x = x0
        segments = self._parse_segments(text)
        
        line_count = 0 

        for seg_type, content in segments:
            if line_count > 0 and current_baseline_y > bottom_limit:
                break
                
            if seg_type == 'text':
                for char in content:
                    if char == '\n':
                        cursor_x = x0
                        current_baseline_y += line_height
                        line_count += 1
                        if current_baseline_y > bottom_limit:
                            break
                        continue
                    w = self.calc_font.text_length(char, fontsize=font_size)
                    
                    # 换行检测
                    if cursor_x + w > rect.x1:
                        cursor_x = x0
                        current_baseline_y += line_height
                        line_count += 1
                        if current_baseline_y > bottom_limit:
                            break 
                    
                    # [新增] 使用提取的文字颜色
                    page.insert_text(
                        (cursor_x, current_baseline_y), 
                        char, 
                        fontname=font_name, 
                        fontfile=font_file, 
                        fontsize=font_size, 
                        color=text_color
                    )
                    cursor_x += w + char_sp
                    
            elif seg_type == 'math':
                latex_content = content.strip('$').strip()
                cursor_x, current_baseline_y, line_count, ok = self._draw_formula_segment(
                    page,
                    rect,
                    cursor_x,
                    current_baseline_y,
                    line_height,
                    font_size,
                    char_sp,
                    latex_content,
                    text_color=text_color,
                    line_count=line_count,
                    bottom_limit=bottom_limit,
                )
                if not ok:
                    break

    # ==================== 数学字体识别 ====================
    
    MATH_FONTS = {'CMMI', 'CMSY', 'CMR', 'CMEX', 'CMSS', 'CMBX', 'CMTI', 'Symbol', 'Math'}
    
    # 数学符号集合（这些字符通常需要用数学字体渲染）
    MATH_SYMBOLS = set('×÷±∓∞∝∑∏∫∮∂∇√∛∜≠≈≡≤≥≪≫⊂⊃⊆⊇∈∉∪∩∧∨¬∀∃∅αβγδεζηθικλμνξπρστυφχψωΓΔΘΛΞΠΣΦΨΩ∙·')
    
    def _is_math_font(self, font_name: str) -> bool:
        """判断是否为数学字体"""
        if not font_name:
            return False
        font_upper = font_name.upper()
        return any(mf in font_upper for mf in self.MATH_FONTS)
    
    def _is_math_symbol(self, char: str) -> bool:
        """判断是否为数学符号"""
        return char in self.MATH_SYMBOLS
    
    def _get_math_font_file(self) -> str:
        """获取数学字体文件路径（使用系统数学字体或回退到默认）"""
        import platform
        
        fonts_dir = os.path.join(self._base_dir, 'fonts')
        
        # 根据操作系统选择字体路径
        if platform.system() == 'Windows':
            # Windows 字体路径
            math_font_candidates = [
                'C:/Windows/Fonts/cambria.ttc',      # Cambria Math
                'C:/Windows/Fonts/times.ttf',        # Times (含部分数学符号)
                'C:/Windows/Fonts/arial.ttf',        # Arial (回退)
            ]
        else:
            # Linux/WSL 字体路径
            math_font_candidates = [
                # 优先使用项目内的字体
                os.path.join(fonts_dir, 'times.ttf'),
                os.path.join(fonts_dir, 'timesbd.ttf'),
                os.path.join(fonts_dir, 'DejaVuSansMono.ttf'),
                # Linux 系统字体路径
                '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',
                '/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf',
                '/usr/share/fonts/truetype/freefont/FreeSans.ttf',
            ]
        
        for path in math_font_candidates:
            if os.path.exists(path):
                return path
        return None
    
    # ==================== 逐 Span 渲染方法 ====================
    
    def draw_rich_content(self, page, bbox, rich_spans: List[Dict], base_style: Dict):
        """
        [V3 新增] 逐 Span 渲染 - 支持数学字体保留
        
        对于数学字体 (CMMI, CMSY, CMR 等)，保留原始字符不映射字体
        对于普通文本，使用中文字体渲染
        
        Args:
            page: PDF 页面对象
            bbox: 边界框
            rich_spans: 富文本 spans 列表，每个包含 text, font, size, flags, color
            base_style: 基础样式字典（用于计算行高等）
        """
        if not rich_spans:
            return
        
        # 清理区域
        bg_color = base_style.get('bg_color', None)
        self._deep_clean_area_with_bg(page, bbox, bg_color)
        
        rect = fitz.Rect(bbox)
        x0, y0 = rect.tl
        
        # 计算基础参数 + [V18] 字号归一化
        base_font_size = self._resolve_font_size(base_style)
        line_height = base_font_size * base_style.get('line', 1.35)
        char_sp = base_style.get('char', 0.0)
        
        # 初始化位置
        ascent = base_font_size * 0.88
        bottom_limit = self._resolve_render_bottom_limit(rect, base_style, base_font_size)
        current_baseline_y = y0 + ascent + (line_height - base_font_size) / 2
        cursor_x = x0
        
        # 遍历每个 span
        for span in rich_spans:
            span_text = span.get('text', '')
            if not span_text:
                continue
            
            span_font = span.get('font', 'SimSun')
            span_size = span.get('size', base_font_size)
            span_flags = span.get('flags', 0)
            span_color = span.get('color', 0)
            original_font = span.get('original_font', span_font)
            
            # 转换颜色
            if isinstance(span_color, int):
                r = ((span_color >> 16) & 0xFF) / 255.0
                g = ((span_color >> 8) & 0xFF) / 255.0
                b = (span_color & 0xFF) / 255.0
                text_color = (r, g, b)
            elif isinstance(span_color, (tuple, list)) and len(span_color) >= 3:
                text_color = tuple(float(c) if c <= 1 else c/255.0 for c in span_color[:3])
            else:
                text_color = (0, 0, 0)
            
            # 预先获取普通字体信息
            is_bold = bool(span_flags & 16)
            is_italic = bool(span_flags & 2)
            font_key = base_style.get('font_key', 'body')
            _, normal_font_file, normal_font_name = self._get_font_for_style(font_key, is_bold, is_italic)
            
            # 【关键】解析 span 中的 LaTeX 公式和普通文本
            segments = self._parse_segments(span_text)
            
            for seg_type, content in segments:
                if seg_type == 'math':
                    cursor_x, current_baseline_y, _, ok = self._draw_formula_segment(
                        page,
                        rect,
                        cursor_x,
                        current_baseline_y,
                        line_height,
                        span_size,
                        char_sp,
                        content.strip('$').strip(),
                        text_color=text_color,
                        line_count=0,
                        bottom_limit=bottom_limit,
                    )
                    if not ok:
                        break
                    continue
                
                # 普通文本渲染
                span_is_math = self._is_math_font(original_font) or self._is_math_font(span_font)
                math_font_file = self._get_math_font_file()
                
                for char in content:
                    if char == '\n':
                        cursor_x = x0
                        current_baseline_y += line_height
                        continue
                    
                    # 计算字符宽度
                    try:
                        w = self.calc_font.text_length(char, fontsize=span_size)
                    except:
                        w = span_size * 0.5  # 回退宽度
                    
                    # 换行检测
                    if cursor_x + w > rect.x1:
                        cursor_x = x0
                        current_baseline_y += line_height
                        if current_baseline_y > bottom_limit:
                            break
                    
                    # 【关键】智能选择字体：数学符号用数学字体，普通字符用中文字体
                    use_math = span_is_math or self._is_math_symbol(char)
                    
                    if use_math and math_font_file:
                        font_file = math_font_file
                        font_name = "math"
                    else:
                        font_file = normal_font_file
                        font_name = normal_font_name
                    
                    # 渲染字符
                    try:
                        page.insert_text(
                            (cursor_x, current_baseline_y),
                            char,
                            fontname=font_name,
                            fontfile=font_file,
                            fontsize=span_size,
                            color=text_color
                        )
                    except Exception as e:
                        # 字体不支持该字符，使用回退字体
                        try:
                            page.insert_text(
                                (cursor_x, current_baseline_y),
                                char,
                                fontsize=span_size,
                                color=text_color
                            )
                        except:
                            pass
                    
                    cursor_x += w + char_sp
        
        return True

    def draw_with_semantic_styles(self, page, bbox, full_text: str, style_mapping: Dict[str, Tuple[bool, bool]], base_style: Dict):
        """
        [语义对齐样式渲染] 使用 LLM 语义对齐的结果进行精准样式渲染
        
        Args:
            page: PDF 页面对象
            bbox: 边界框
            full_text: 完整译文
            style_mapping: {译文中的词: (is_bold, is_italic)}
            base_style: 基础样式
        """
        if not full_text:
            return
        
        # === 步骤1: 标记每个字符的样式 ===
        char_styles = [(False, False)] * len(full_text)
        
        for keyword, (is_bold, is_italic) in style_mapping.items():
            if not keyword:
                continue
            # 在译文中查找所有出现位置
            start = 0
            while True:
                pos = full_text.find(keyword, start)
                if pos == -1:
                    break
                # 标记样式
                for i in range(pos, pos + len(keyword)):
                    if i < len(char_styles):
                        old_bold, old_italic = char_styles[i]
                        char_styles[i] = (old_bold or is_bold, old_italic or is_italic)
                start = pos + 1
        
        # === 步骤2: 渲染 ===
        bg_color = base_style.get('bg_color', None)
        self._deep_clean_area_with_bg(page, bbox, bg_color)
        
        rect = fitz.Rect(bbox)
        x0, y0 = rect.tl
        
        font_size = self._resolve_font_size(base_style)
        line_height = font_size * base_style.get('line', 1.35)
        char_sp = base_style.get('char', 0.0)
        font_key = base_style.get('font_key', 'body')
        text_color = base_style.get('text_color', (0, 0, 0))
        
        if text_color and isinstance(text_color, (tuple, list)) and len(text_color) >= 3:
            text_color = tuple(float(c) for c in text_color[:3])
            if any(c > 1.0 for c in text_color):
                text_color = tuple(c / 255.0 for c in text_color)
        else:
            text_color = (0, 0, 0)
        
        ascent = font_size * 0.88
        bottom_limit = self._resolve_render_bottom_limit(rect, base_style, font_size)
        is_likely_single_line = len(full_text) * font_size < rect.width * 1.5
        start_y_offset = (line_height - font_size) / 2
        
        if is_likely_single_line and rect.height < line_height:
            current_baseline_y = y0 + (rect.height - font_size) / 2 + ascent
        else:
            current_baseline_y = y0 + ascent + start_y_offset
        
        cursor_x = x0
        line_count = 0
        
        # 解析公式
        segments = self._parse_segments(full_text)
        char_idx = 0
        
        for seg_type, content in segments:
            if line_count > 0 and current_baseline_y > bottom_limit:
                break
            
            if seg_type == 'math':
                cursor_x, current_baseline_y, line_count, ok = self._draw_formula_segment(
                    page,
                    rect,
                    cursor_x,
                    current_baseline_y,
                    line_height,
                    font_size,
                    char_sp,
                    content.strip('$').strip(),
                    text_color=text_color,
                    line_count=line_count,
                    bottom_limit=bottom_limit,
                )
                if not ok:
                    break
                
                char_idx += len(content)
                
            elif seg_type == 'text':
                for char in content:
                    if char == '\n':
                        cursor_x = x0
                        current_baseline_y += line_height
                        line_count += 1
                        char_idx += 1
                        if current_baseline_y > bottom_limit:
                            break
                        continue
                    if char_idx < len(char_styles):
                        is_bold, is_italic = char_styles[char_idx]
                    else:
                        is_bold, is_italic = False, False
                    
                    _, font_file, font_name = self._get_font_for_style(font_key, is_bold, is_italic)
                    
                    w = self.calc_font.text_length(char, fontsize=font_size)
                    
                    if cursor_x + w > rect.x1:
                        cursor_x = x0
                        current_baseline_y += line_height
                        line_count += 1
                        if current_baseline_y > bottom_limit:
                            break
                    
                    try:
                        page.insert_text(
                            (cursor_x, current_baseline_y),
                            char,
                            fontname=font_name,
                            fontfile=font_file,
                            fontsize=font_size,
                            color=text_color
                        )
                    except:
                        try:
                            page.insert_text((cursor_x, current_baseline_y), char, fontsize=font_size, color=text_color)
                        except:
                            pass
                    
                    cursor_x += w + char_sp
                    char_idx += 1
        
        return True

    def draw_with_style_keywords(self, page, bbox, full_text: str, orig_rich_spans: List[Dict], base_style: Dict):
        """
        [精准样式渲染 V3] 从原文 rich_spans 提取样式关键词，在译文中精确匹配
        
        策略：
        1. 从原文 rich_spans 提取有特殊样式（粗体/斜体）的词
        2. 这些词可能在译文中保留（如 Figure 1, Table 2, α, β 等）
        3. 在译文中精确匹配这些词并应用样式
        
        Args:
            page: PDF 页面对象
            bbox: 边界框
            full_text: 完整译文
            orig_rich_spans: 原文的 rich_spans（包含样式信息）
            base_style: 基础样式
        """
        if not full_text:
            return
        
        # === 步骤1: 提取原文中的样式关键词 ===
        style_keywords = []  # [(keyword, is_bold, is_italic)]
        
        for span in orig_rich_spans:
            text = span.get('text', '').strip()
            if not text:
                continue
            
            flags = span.get('flags', 0)
            is_bold = bool(flags & 16)
            is_italic = bool(flags & 2)
            
            # 只收集有特殊样式的词
            if is_bold or is_italic:
                # 清理并分词（对于多词的 span）
                words = text.split()
                for word in words:
                    word = word.strip('.,;:!?()[]{}"\' ')
                    if len(word) >= 1:
                        style_keywords.append((word, is_bold, is_italic))
        
        # === 步骤2: 在译文中标记样式位置 ===
        # 每个字符的样式: (is_bold, is_italic)
        char_styles = [(False, False)] * len(full_text)
        
        for keyword, is_bold, is_italic in style_keywords:
            # 在译文中查找该关键词
            start = 0
            while True:
                pos = full_text.find(keyword, start)
                if pos == -1:
                    break
                # 标记这些字符的样式
                for i in range(pos, pos + len(keyword)):
                    if i < len(char_styles):
                        old_bold, old_italic = char_styles[i]
                        char_styles[i] = (old_bold or is_bold, old_italic or is_italic)
                start = pos + 1
        
        # === 步骤3: 渲染（与 draw_content 类似，但按字符应用样式） ===
        bg_color = base_style.get('bg_color', None)
        self._deep_clean_area_with_bg(page, bbox, bg_color)
        
        rect = fitz.Rect(bbox)
        x0, y0 = rect.tl
        
        font_size = self._resolve_font_size(base_style)
        line_height = font_size * base_style.get('line', 1.35)
        char_sp = base_style.get('char', 0.0)
        font_key = base_style.get('font_key', 'body')
        text_color = base_style.get('text_color', (0, 0, 0))
        
        # 确保颜色格式正确
        if text_color and isinstance(text_color, (tuple, list)) and len(text_color) >= 3:
            text_color = tuple(float(c) for c in text_color[:3])
            if any(c > 1.0 for c in text_color):
                text_color = tuple(c / 255.0 for c in text_color)
        else:
            text_color = (0, 0, 0)
        
        ascent = font_size * 0.88
        bottom_limit = self._resolve_render_bottom_limit(rect, base_style, font_size)
        is_likely_single_line = len(full_text) * font_size < rect.width * 1.5
        start_y_offset = (line_height - font_size) / 2
        
        if is_likely_single_line and rect.height < line_height:
            current_baseline_y = y0 + (rect.height - font_size) / 2 + ascent
        else:
            current_baseline_y = y0 + ascent + start_y_offset
        
        cursor_x = x0
        line_count = 0
        
        # 解析公式
        segments = self._parse_segments(full_text)
        char_idx = 0
        
        for seg_type, content in segments:
            if line_count > 0 and current_baseline_y > bottom_limit:
                break
            
            if seg_type == 'math':
                cursor_x, current_baseline_y, line_count, ok = self._draw_formula_segment(
                    page,
                    rect,
                    cursor_x,
                    current_baseline_y,
                    line_height,
                    font_size,
                    char_sp,
                    content.strip('$').strip(),
                    text_color=text_color,
                    line_count=line_count,
                    bottom_limit=bottom_limit,
                )
                if not ok:
                    break
                
                char_idx += len(content)
                
            elif seg_type == 'text':
                for char in content:
                    if char == '\n':
                        cursor_x = x0
                        current_baseline_y += line_height
                        line_count += 1
                        char_idx += 1
                        if current_baseline_y > bottom_limit:
                            break
                        continue
                    # 获取当前字符的样式
                    if char_idx < len(char_styles):
                        is_bold, is_italic = char_styles[char_idx]
                    else:
                        is_bold, is_italic = False, False
                    
                    # 获取字体
                    _, font_file, font_name = self._get_font_for_style(font_key, is_bold, is_italic)
                    
                    w = self.calc_font.text_length(char, fontsize=font_size)
                    
                    if cursor_x + w > rect.x1:
                        cursor_x = x0
                        current_baseline_y += line_height
                        line_count += 1
                        if current_baseline_y > bottom_limit:
                            break
                    
                    try:
                        page.insert_text(
                            (cursor_x, current_baseline_y),
                            char,
                            fontname=font_name,
                            fontfile=font_file,
                            fontsize=font_size,
                            color=text_color
                        )
                    except:
                        try:
                            page.insert_text((cursor_x, current_baseline_y), char, fontsize=font_size, color=text_color)
                        except:
                            pass
                    
                    cursor_x += w + char_sp
                    char_idx += 1
        
        return True

    def draw_rich_content_v2(self, page, bbox, full_text: str, rich_spans: List[Dict], base_style: Dict):
        """
        [V2 精准样式渲染] 结合完整文本的公式解析和 rich_spans 的样式信息
        
        解决的问题：
        1. 公式被拆分到多个 span 中导致无法渲染
        2. 整段使用同一样式导致精准样式丢失
        
        方案：
        1. 使用完整文本解析公式位置
        2. 构建字符位置到样式的映射
        3. 按样式分段渲染，公式用图片
        
        Args:
            page: PDF 页面对象
            bbox: 边界框
            full_text: 完整的译文（用于公式解析）
            rich_spans: 富文本样式信息列表
            base_style: 基础样式字典
        """
        if not full_text:
            return
        
        # 清理区域
        bg_color = base_style.get('bg_color', None)
        self._deep_clean_area_with_bg(page, bbox, bg_color)
        
        rect = fitz.Rect(bbox)
        x0, y0 = rect.tl
        
        # 基础参数 + [V18] 字号归一化
        base_font_size = self._resolve_font_size(base_style)
        line_height = base_font_size * base_style.get('line', 1.35)
        char_sp = base_style.get('char', 0.0)
        font_key = base_style.get('font_key', 'body')
        default_color = base_style.get('text_color', (0, 0, 0))
        
        # [V17] 保持LaTeX原样，交给图片渲染
        # full_text = convert_latex_to_unicode(full_text)  # 禁用
        
        # === 步骤1: 构建字符位置到样式的映射 ===
        # 将 rich_spans 的文本拼接，建立每个字符的样式映射
        char_styles = []  # 每个字符的样式: (is_bold, is_italic, color)
        
        if rich_spans:
            for span in rich_spans:
                span_text = span.get('text', '')
                flags = span.get('flags', 0)
                is_bold = bool(flags & 16)
                is_italic = bool(flags & 2)
                span_color = span.get('color', 0)
                
                # 转换颜色
                if isinstance(span_color, int):
                    r = ((span_color >> 16) & 0xFF) / 255.0
                    g = ((span_color >> 8) & 0xFF) / 255.0
                    b = (span_color & 0xFF) / 255.0
                    color = (r, g, b)
                elif isinstance(span_color, (tuple, list)) and len(span_color) >= 3:
                    color = tuple(float(c) if c <= 1 else c/255.0 for c in span_color[:3])
                else:
                    color = default_color
                
                for _ in span_text:
                    char_styles.append((is_bold, is_italic, color))
        
        # === 步骤2: 解析公式位置 ===
        segments = self._parse_segments(full_text)
        
        # === 步骤3: 渲染 ===
        ascent = base_font_size * 0.88
        bottom_limit = self._resolve_render_bottom_limit(rect, base_style, base_font_size)
        is_likely_single_line = len(full_text) * base_font_size < rect.width * 1.5
        start_y_offset = (line_height - base_font_size) / 2
        
        if is_likely_single_line and rect.height < line_height:
            current_baseline_y = y0 + (rect.height - base_font_size) / 2 + ascent
        else:
            current_baseline_y = y0 + ascent + start_y_offset
        
        cursor_x = x0
        char_idx = 0  # 当前字符在 full_text 中的位置
        line_count = 0
        
        for seg_type, content in segments:
            if line_count > 0 and current_baseline_y > bottom_limit:
                break
            
            if seg_type == 'math':
                cursor_x, current_baseline_y, line_count, ok = self._draw_formula_segment(
                    page,
                    rect,
                    cursor_x,
                    current_baseline_y,
                    line_height,
                    base_font_size,
                    char_sp,
                    content.strip('$').strip(),
                    text_color=default_color,
                    line_count=line_count,
                    bottom_limit=bottom_limit,
                )
                if not ok:
                    break
                
                # 更新字符索引
                char_idx += len(content)
                
            elif seg_type == 'text':
                # 文本渲染，使用精准样式
                for char in content:
                    if char == '\n':
                        cursor_x = x0
                        current_baseline_y += line_height
                        char_idx += 1
                        if current_baseline_y > bottom_limit:
                            break
                        continue
                    # 获取当前字符的样式
                    if char_idx < len(char_styles):
                        is_bold, is_italic, text_color = char_styles[char_idx]
                    else:
                        is_bold, is_italic, text_color = False, False, default_color
                    
                    # 获取字体
                    _, font_file, font_name = self._get_font_for_style(font_key, is_bold, is_italic)
                    
                    # 计算宽度
                    w = self.calc_font.text_length(char, fontsize=base_font_size)
                    
                    # 换行检测
                    if cursor_x + w > rect.x1:
                        cursor_x = x0
                        current_baseline_y += line_height
                        line_count += 1
                        if current_baseline_y > bottom_limit:
                            break 
                    
                    # 渲染字符
                    try:
                        page.insert_text(
                            (cursor_x, current_baseline_y),
                            char,
                            fontname=font_name,
                            fontfile=font_file,
                            fontsize=base_font_size,
                            color=text_color
                        )
                    except:
                        try:
                            page.insert_text(
                                (cursor_x, current_baseline_y),
                                char,
                                fontsize=base_font_size,
                                color=text_color
                            )
                        except:
                            pass
                    
                    cursor_x += w + char_sp
                    char_idx += 1
        
        return True
