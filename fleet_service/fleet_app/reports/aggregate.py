"""
Tính toán cho báo cáo so sánh — TÁCH KHỎI phần dựng HTML.

Vì sao tách: mọi con số trong báo cáo phải kiểm được bằng một hàm gọi trực tiếp,
không phải bằng cách đọc chuỗi HTML. Bản trước trộn hai việc, nên câu duy nhất
trả lời được "tuần này tính từ đâu tới đâu" là đọc f-string.

Không có phụ thuộc nào ngoài stdlib ở đây — không matplotlib, không WeasyPrint.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

# Ngưỡng tỉ lệ đạt, dùng chung cho màu ô KPI và chữ trạng thái.
RATE_GOOD, RATE_WATCH = 95.0, 85.0

# ═══════════════════════════════════════════════════════════════════════════
# Ngôn ngữ
# ═══════════════════════════════════════════════════════════════════════════
# Một bảng chữ duy nhất cho CẢ BA module của báo cáo (tính, biểu đồ, dựng
# trang). Bản trước chỉ có tiếng Việt nướng thẳng vào f-string, nên báo cáo
# không gửi được cho khách hoặc kiểm toán nước ngoài — mà đó lại đúng là loại
# người đọc cần một bản in.
#
# Đặt ở module TÍNH chứ không ở module dựng trang, vì các câu trong `findings()`
# cũng là chữ người đọc, và chúng sinh ra ở đây.
LANG = "vi"


def set_lang(lang: str) -> None:
    """Gọi một lần ở đầu mỗi lần render. Bất cứ thứ gì không phải 'en' là 'vi'."""
    global LANG
    LANG = "en" if str(lang or "").lower().startswith("en") else "vi"


TEXT = {
 "vi": {
  "title_compare":   "BÁO CÁO SO SÁNH DÂY CHUYỀN",
  "title_fp":        "VÂN TAY KIỂU LỖI",
  "sub":             "{n} dây chuyền · {period} · nhà máy đóng gói gia vị",
  "generated":       "Lập lúc {at}",
  "logo":            "logo",
  "cov_ok":          "Đủ cả {ok}/{total} máy.",
  "cov_ok_more":     "Mọi con số dưới đây đọc từ chính các máy tại thời điểm lập báo cáo.",
  "cov_bad":         "KHÔNG đủ đội hình — chỉ có số của {ok}/{total} máy.",
  "cov_bad_more":    ("Các máy thiếu vẫn có mặt trong báo cáo kèm lý do; con số tổng ở "
                      "dưới là tổng của {ok} máy, không phải của cả nhà máy."),
  "cov_none":        "KHÔNG đọc được máy nào.",
  "cov_none_more":   ("Báo cáo này rỗng vì tầng fleet không thấy máy nào trong tailnet, "
                      "KHÔNG phải vì nhà máy không sản xuất. Kiểm tra Tailscale và tiến "
                      "trình fleet service rồi lập lại."),
  "kpi_total":       "Tổng sản lượng", "kpi_pass": "Đạt", "kpi_fail": "Không đạt",
  "kpi_rate":        "Tỉ lệ đạt chung",
  "kpi_perday":      "{n} sản phẩm/ngày toàn nhà máy",
  "kpi_thresh":      "{word} (ngưỡng {t})",
  "rate_good": "trong ngưỡng", "rate_watch": "cần theo dõi",
  "rate_bad": "dưới ngưỡng", "rate_none": "không có số",
  "h_rhythm":        "Nhịp sản xuất theo ngày",
  "lead_rhythm":     ("Cột chồng theo máy, nên một ngày sản lượng tụt là đọc được ngay "
                      "máy nào tụt — thứ mà bảng dưới không nói được."),
  "h_bymachine":     "Sản lượng theo máy",
  "th_machine": "Máy", "th_output": "Sản lượng",
  "th_perday_period": "Mỗi ngày<br>(cả kỳ)", "th_perday_active": "Mỗi ngày<br>(có chạy)",
  "th_days": "Ngày<br>có chạy", "th_rate": "Tỉ lệ đạt", "th_recipe": "Recipe chính",
  "th_total": "TỔNG", "no_data_period": "không có dữ liệu trong kỳ",
  "note_norank":     ("Không xếp hạng theo tỉ lệ đạt: các máy chạy recipe khác nhau, nên "
                      "tỉ lệ đạt phản ánh độ khó của mặt hàng chứ không phản ánh máy. Hai "
                      "cột “mỗi ngày” cố ý đặt cạnh nhau — máy chạy ít ngày trong kỳ thì "
                      "cột đầu thấp hơn năng lực thật."),
  "h_fp":            "Tỉ trọng nguyên nhân trên mẫu fail",
  "lead_fp":         ("Các ô của MỘT máy cộng lại bằng 100. Đây là cách so sánh có nghĩa "
                      "giữa các máy, vì những nguyên nhân này thuộc pipeline OCR chứ không "
                      "thuộc mặt hàng — khác với tỉ lệ đạt, thứ phản ánh độ khó của sản "
                      "phẩm đang chạy."),
  "fp_none":         "Không có mẫu fail nào trong kỳ để dựng vân tay kiểu lỗi.",
  "th_sample": "Mẫu", "th_median": "Trung vị",
  "sample_all": "phủ hết kỳ", "sample_part": "lấy mẫu",
  "note_fp":         ("Tên cột viết ngắn; tên đầy đủ ở chú giải biểu đồ phía trên, nối "
                      "bằng ô màu. Ô tô đỏ = cao hơn trung vị của cột từ 12 điểm trở lên. "
                      "Trung vị chỉ tính khi có từ {n} máy trở lên cùng có số cho cột đó — "
                      "ít hơn thì “trung vị” là chính con số duy nhất ấy, nên để “—”. Cột "
                      "“Mẫu” ghi cỡ mẫu của từng máy: “lấy mẫu” nghĩa là các tỉ trọng trên "
                      "hàng đó là tỉ trọng CỦA MẪU, không phải của toàn bộ số fail trong kỳ."),
  "h_howto":         "Đọc vân tay thế nào",
  "howto_nodet":     ("<b>Detector không thấy vùng nào</b> cao → đi xem camera, trigger, "
                      "ánh sáng. Máy không nhìn thấy chỗ cần đọc, nên chưa tới bước đọc."),
  "howto_char":      ("<b>Ký tự dưới ngưỡng tin cậy</b> cao → máy THẤY vùng in nhưng không "
                      "đủ tự tin về ký tự: nét in, mực, tiêu cự."),
  "howto_text":      ("<b>OCR đọc sai chuỗi</b> cao → đọc rõ nhưng ra chuỗi khác chuỗi mong "
                      "đợi: xem lại chuỗi khai trong recipe."),
  "howto_tmpl":      ("<b>Ảnh không khớp template</b> cao → template có thể đã cũ so với bao "
                      "bì đang chạy."),
  "howto_close":     ("Hai máy cùng “tỉ lệ đạt thấp” có thể đang hỏng hai thứ hoàn toàn khác "
                      "nhau — đó là lý do trang này tồn tại, và là thứ duy nhất so được giữa "
                      "các máy chạy khác mặt hàng."),
  "h_findings":      "Phát hiện chính",
  "h_norm":          "Sản lượng chuẩn hoá theo máy",
  "lead_norm":       ("Mỗi máy hai cột: chia cho cả kỳ, và chia cho số ngày thực có chạy. "
                      "Máy chạy 2 trong 7 ngày mà chỉ đọc cột thứ nhất thì trông như máy "
                      "yếu — đọc cả hai mới ra “chạy nhanh nhưng chạy ít ngày”. Đây là câu "
                      "trả lời cho “vì sao máy này thấp”, nên nó đứng ngay trước phần chi "
                      "tiết từng máy."),
  "h_appendix":      "Phụ lục theo máy · chia theo tuần",
  "lead_appendix":   ("Mỗi máy một thẻ, chia theo tuần ISO (thứ Hai đầu tuần) — không phải "
                      "cửa sổ 7 ngày trượt, để hai báo cáo lập cách nhau một ngày vẫn so "
                      "được với nhau. Ngày đứng máy trong tuần được ghi ngay dưới tên tuần. "
                      "Kỳ chỉ có một tuần thì biểu đồ vẽ theo ngày, vì một cột đơn độc không "
                      "so được với gì cả."),
  "card_output": "Sản lượng", "card_rate": "Tỉ lệ đạt", "card_fail": "Không đạt",
  "card_peractive": "{n}/ngày có chạy", "card_on": "trên {n} sản phẩm",
  "no_device": "thiết bị chưa khai",
  "card_nodata":     "Không có dữ liệu trong kỳ.",
  "card_nodata_more": ("máy có thể vẫn đang sản xuất; thiếu ở đây là thiếu ĐƯỜNG ĐỌC SỐ, "
                       "không phải thiếu sản lượng."),
  "th_week": "Tuần", "th_wpass": "Đạt", "th_wfail": "Không đạt",
  "th_wperactive": "Mỗi ngày<br>có chạy", "th_delta": "So tuần trước",
  "idle_days": " · {n} ngày không chạy",
  "week_none": "Không có ngày nào có sản lượng trong kỳ.",
  "pts": "điểm",
  "recipes_in": "Recipe trong kỳ: {list}",
  "top_cause_line": "Nguyên nhân lớn nhất trên mẫu: <b>{label}</b> ({pct} của mẫu{sampling}).",
  "worst_day": "Ngày thấp nhất: {d} — {rate} trên {n} sản phẩm.",
  "few_days": "Chỉ có sản lượng {a}/{b} ngày của kỳ.",
  "foot":            ("Sinh bởi Fleet Service · dữ liệu đọc trực tiếp từ từng máy tại thời "
                      "điểm lập báo cáo · giá trị không đo được hiện “—”, không hiện 0."),
  "page_of": "Trang {p} / {n}",
  # findings
  "f_missing":       ("{m} không có số trong kỳ này — báo cáo thiếu một dây chuyền, không "
                      "phải dây chuyền đó không sản xuất."),
  "f_missing_act":   "kiểm tra agent trên máy đó rồi lập lại báo cáo",
  "f_below":         ("{m} đạt {rate} — dưới ngưỡng {t}. Nguyên nhân chiếm tỉ trọng lớn "
                      "nhất trên mẫu: {label} ({pct} của mẫu)."),
  "f_slip":          "{m} tuần {w} giảm {d} điểm so với tuần trước ({a} → {b}).",
  "f_slip_act":      "xem phụ lục theo tuần của máy này để biết trượt từ ngày nào",
  "f_fewdays":       ("Chạy dưới nửa số ngày trong kỳ: {names}. Cột “mỗi ngày” chia cho cả "
                      "kỳ nên thấp hơn năng lực thật."),
  "f_fewdays_act":   "đọc cột “mỗi ngày có chạy” bên cạnh để so năng lực",
  "f_fewdays_item":  "{m} ({a}/{b} ngày)",
  "f_outlier":       "{m} lệch hẳn khỏi phần còn lại ở “{label}”: {pct} của mẫu so với trung vị {med}.",
  "f_outlier_act":   "so với máy chạy cùng recipe",
  "f_none":          "Không có máy nào dưới ngưỡng, không có tuần nào trượt quá 3 điểm.",
  "f_none_act":      "không cần hành động",
  "f_detail_act":    "xem chi tiết ở phụ lục máy",
  # charts
  "c_perday": "sản phẩm / ngày", "c_products": "sản phẩm",
  "c_period": "chia cho cả kỳ", "c_active": "chia cho ngày có chạy",
  "c_share": "tỉ trọng giữa các nguyên nhân, trên mẫu fail (%)",
  "c_nosample": "không có mẫu trong kỳ",
  "c_pass": "đạt", "c_fail": "không đạt", "c_rate": "tỉ lệ đạt", "c_rate_ax": "tỉ lệ đạt (%)",
  "week_short": "Tuần {n}",
  # Excel / CSV
  "sh_output": "Sản lượng", "sh_weekly": "Theo tuần", "sh_fp": "Vân tay kiểu lỗi",
  "x_machine": "Máy", "x_line": "Dây chuyền", "x_device": "Thiết bị",
  "x_output": "Sản lượng", "x_perday_period": "Mỗi ngày (cả kỳ)",
  "x_perday_active": "Mỗi ngày (có chạy)", "x_days": "Ngày có chạy",
  "x_pass": "Đạt", "x_fail": "Không đạt", "x_rate_pct": "Tỉ lệ đạt (%)",
  "x_recipe": "Recipe chính", "x_note": "Ghi chú",
  "x_isoweek": "Tuần ISO", "x_span": "Từ–đến", "x_full7": "Đủ 7 ngày",
  "x_delta_pts": "So tuần trước (điểm)", "x_samplesize": "Cỡ mẫu",
  "x_coversall": "Phủ hết kỳ", "x_none_week": "Không có tuần nào có sản lượng trong kỳ.",
  "x_fp_note": ("Các ô là TỈ TRỌNG giữa các nguyên nhân (một máy cộng lại = 100), "
                "không phải % sản phẩm trượt từng bước."),
  "x_block1": "# Tổng hợp theo máy", "x_block2": "# Chi tiết theo tuần",
  "reason_unknown": "không rõ lý do",
 },
 "en": {
  "title_compare":   "PACKING LINE COMPARISON",
  "title_fp":        "FAILURE FINGERPRINT",
  "sub":             "{n} packing lines · {period} · spice packing plant",
  "generated":       "Generated {at}",
  "logo":            "logo",
  "cov_ok":          "All {ok}/{total} lines reported.",
  "cov_ok_more":     "Every figure below was read from the lines themselves at report time.",
  "cov_bad":         "INCOMPLETE — only {ok}/{total} lines reported.",
  "cov_bad_more":    ("The missing lines still appear below with their reason; the totals are "
                      "the total of {ok} lines, not of the plant."),
  "cov_none":        "NO line could be read.",
  "cov_none_more":   ("This report is empty because the fleet layer saw no machine on the "
                      "tailnet — NOT because the plant produced nothing. Check Tailscale and "
                      "the fleet service process, then generate it again."),
  "kpi_total":       "Total output", "kpi_pass": "Pass", "kpi_fail": "Fail",
  "kpi_rate":        "Overall pass rate",
  "kpi_perday":      "{n} products/day across the plant",
  "kpi_thresh":      "{word} (threshold {t})",
  "rate_good": "within threshold", "rate_watch": "needs watching",
  "rate_bad": "below threshold", "rate_none": "no reading",
  "h_rhythm":        "Daily production rhythm",
  "lead_rhythm":     ("Stacked by line, so a day that dips tells you immediately which line "
                      "dipped — something the table below cannot."),
  "h_bymachine":     "Output by line",
  "th_machine": "Line", "th_output": "Output",
  "th_perday_period": "Per day<br>(whole period)", "th_perday_active": "Per day<br>(days run)",
  "th_days": "Days<br>run", "th_rate": "Pass rate", "th_recipe": "Main product",
  "th_total": "TOTAL", "no_data_period": "no data in this period",
  "note_norank":     ("No ranking by pass rate: the lines run different products, so pass "
                      "rate reflects how hard the product is to read, not how good the line "
                      "is. The two “per day” columns sit side by side on purpose — a line "
                      "that ran few days in the period looks weaker than it is in the first."),
  "h_fp":            "Share of failure causes, on the fail sample",
  "lead_fp":         ("The cells of ONE line sum to 100. This is the comparison that means "
                      "something across lines, because these causes belong to the OCR "
                      "pipeline and not to the product — unlike pass rate, which reflects "
                      "how hard the product running is."),
  "fp_none":         "No fail sample in this period, so there is no fingerprint to draw.",
  "th_sample": "Sample", "th_median": "Median",
  "sample_all": "whole period", "sample_part": "sampled",
  "note_fp":         ("Column names are shortened; the full names are in the chart legend "
                      "above, tied by the colour chip. A red cell is at least 12 points above "
                      "the column median. The median is only computed when {n} or more lines "
                      "have a value for that column — with fewer, the “median” is that single "
                      "figure itself, so it prints “—”. The “Sample” column carries each "
                      "line's sample size: “sampled” means the shares on that row are shares "
                      "OF THE SAMPLE, not of every failure in the period."),
  "h_howto":         "How to read the fingerprint",
  "howto_nodet":     ("High <b>detector found no region</b> → go and look at the camera, the "
                      "trigger, the lighting. The machine cannot see the printed area, so it "
                      "never reaches the reading step."),
  "howto_char":      ("High <b>characters below confidence threshold</b> → the machine DOES "
                      "see the print but is not confident about the characters: print quality, "
                      "ink, focus."),
  "howto_text":      ("High <b>OCR read the wrong string</b> → read clearly but came out as a "
                      "different string than expected: check the expected string in the recipe."),
  "howto_tmpl":      ("High <b>image does not match the template</b> → the template may be out "
                      "of date against the packaging currently running."),
  "howto_close":     ("Two lines with the same low pass rate can be broken in completely "
                      "different ways — that is why this page exists, and it is the only "
                      "comparison that holds across lines running different products."),
  "h_findings":      "Key findings",
  "h_norm":          "Normalised output by line",
  "lead_norm":       ("Two bars per line: divided by the whole period, and divided by the days "
                      "the line actually ran. A line that ran 2 of 7 days looks weak if you "
                      "read only the first — read both and it says “fast, but ran few days”. "
                      "This answers “why is this line low?”, which is why it sits immediately "
                      "before the per-line detail."),
  "h_appendix":      "Per-line appendix · by week",
  "lead_appendix":   ("One card per line, split by ISO week (Monday first) — not a rolling "
                      "7-day window, so two reports generated a day apart still compare. Idle "
                      "days inside a week are noted under the week name. When the period spans "
                      "one week the chart is drawn by day, because a single bar has nothing to "
                      "compare against."),
  "card_output": "Output", "card_rate": "Pass rate", "card_fail": "Fail",
  "card_peractive": "{n}/day run", "card_on": "of {n} products",
  "no_device": "device not declared",
  "card_nodata":     "No data in this period.",
  "card_nodata_more": ("the line may well still be packing; what is missing here is the WAY TO "
                       "READ IT, not the output."),
  "th_week": "Week", "th_wpass": "Pass", "th_wfail": "Fail",
  "th_wperactive": "Per day<br>run", "th_delta": "vs previous week",
  "idle_days": " · {n} idle days",
  "week_none": "No day in this period produced anything.",
  "pts": "pts",
  "recipes_in": "Products in period: {list}",
  "top_cause_line": "Largest share of the sample: <b>{label}</b> ({pct} of the sample{sampling}).",
  "worst_day": "Weakest day: {d} — {rate} on {n} products.",
  "few_days": "Only produced on {a} of {b} days in the period.",
  "foot":            ("Generated by Fleet Service · read directly from each line at report "
                      "time · a value that could not be measured prints “—”, never 0."),
  "page_of": "Page {p} / {n}",
  "f_missing":       ("{m} has no figures for this period — the report is missing a line, not "
                      "reporting a line that produced nothing."),
  "f_missing_act":   "check the agent on that machine, then generate the report again",
  "f_below":         ("{m} is at {rate} — below the {t} threshold. Largest share of its fail "
                      "sample: {label} ({pct} of the sample)."),
  "f_slip":          "{m} dropped {d} pts in week {w} against the week before ({a} → {b}).",
  "f_slip_act":      "see this line's weekly appendix to find the day it started",
  "f_fewdays":       ("Ran on fewer than half the days in the period: {names}. The “per day” "
                      "column divides by the whole period, so it reads lower than the real rate."),
  "f_fewdays_act":   "read the “per day run” column beside it to compare capability",
  "f_fewdays_item":  "{m} ({a}/{b} days)",
  "f_outlier":       "{m} stands well clear of the rest on “{label}”: {pct} of the sample against a median of {med}.",
  "f_outlier_act":   "compare against a line running the same product",
  "f_none":          "No line is below threshold and no week slipped by more than 3 points.",
  "f_none_act":      "no action needed",
  "f_detail_act":    "see the per-line appendix",
  "c_perday": "products / day", "c_products": "products",
  "c_period": "per period day", "c_active": "per day run",
  "c_share": "share between causes, on the fail sample (%)",
  "c_nosample": "no sample in this period",
  "c_pass": "pass", "c_fail": "fail", "c_rate": "pass rate", "c_rate_ax": "pass rate (%)",
  "week_short": "Week {n}",
  "sh_output": "Output", "sh_weekly": "By week", "sh_fp": "Failure fingerprint",
  "x_machine": "Line", "x_line": "Packing line", "x_device": "Device",
  "x_output": "Output", "x_perday_period": "Per day (whole period)",
  "x_perday_active": "Per day (days run)", "x_days": "Days run",
  "x_pass": "Pass", "x_fail": "Fail", "x_rate_pct": "Pass rate (%)",
  "x_recipe": "Main product", "x_note": "Note",
  "x_isoweek": "ISO week", "x_span": "From–to", "x_full7": "Full 7 days",
  "x_delta_pts": "vs previous week (pts)", "x_samplesize": "Sample size",
  "x_coversall": "Whole period", "x_none_week": "No week in this period produced anything.",
  "x_fp_note": ("Cells are the SHARE between causes (one line sums to 100), not the "
                "percentage of products failing each step."),
  "x_block1": "# Summary by line", "x_block2": "# Weekly detail",
  "reason_unknown": "reason unknown",
 },
}


def T(key: str, **kw) -> str:
    s = TEXT.get(LANG, TEXT["vi"]).get(key) or TEXT["vi"].get(key, key)
    return s.format(**kw) if kw else s


# Nhãn nguyên nhân tiếng Anh, khoá theo MÃ bước kiểm tra. Edge chỉ trả nhãn
# tiếng Việt, nên bản EN phải tra ở đây chứ không lấy từ payload.
CAUSE_EN = {
    "char_verification": "Characters below confidence threshold",
    "no_detection": "Detector found no region in frame",
    "template_verification": "Image does not match the template",
    "text_verification": "OCR read the wrong string",
    "product_verification": "Product not recognised",
}
CAUSE_SHORT_EN = {
    "char_verification": "Characters", "no_detection": "No region",
    "text_verification": "Wrong string", "template_verification": "Template",
    "product_verification": "Product",
}


def cause_label(cause: str, fallback: str = "") -> str:
    """Nhãn người đọc cho một mã nguyên nhân, theo ngôn ngữ đang chọn."""
    if LANG == "en":
        return CAUSE_EN.get(cause) or fallback or cause
    return fallback or cause


def fmt_num(v: Optional[float], d: int = 0, dash: str = "—") -> str:
    """Dấu nghìn "." và dấu thập phân "," — lối viết số ở đây.

    Định nghĩa nằm ở module này chứ không ở lớp vẽ, vì các câu trong `findings()`
    cũng chứa số và phải viết giống hệt bảng ngay bên trên nó. Bản đầu để lớp vẽ
    tự định dạng, nên bảng in "63,88%" còn phát hiện chính in "63.88%" — hai lối
    viết trong cùng một trang.
    """
    if v is None:
        return dash
    raw = f"{v:,.{d}f}"
    if LANG == "en":
        return raw                      # 126,587.5 — lối viết Anh/Mỹ
    return raw.replace(",", "\x00").replace(".", ",").replace("\x00", ".")


def fmt_pct(v: Optional[float], d: int = 2) -> str:
    return "—" if v is None else f"{fmt_num(v, d)}%"


def rate_level(rate: Optional[float]) -> str:
    """'none' khi không đo được — KHÔNG phải 'bad'.

    Trộn hai thứ này là cách nhanh nhất để một máy tắt agent trông như một máy
    đang chạy hỏng: cả hai ra ô đỏ, mà việc cần làm thì khác hẳn nhau.
    """
    if rate is None:
        return "none"
    if rate >= RATE_GOOD:
        return "good"
    if rate >= RATE_WATCH:
        return "watch"
    return "bad"


def _d(key: str) -> Optional[date]:
    try:
        return datetime.strptime(key[:10], "%Y-%m-%d").date()
    except (ValueError, TypeError):
        return None


def daily(trend: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Chuỗi ngày đã sắp, mỗi phần tử có pass/fail/total/rate."""
    out = []
    for k in sorted(trend or {}):
        v = (trend or {})[k] or {}
        p, f = v.get("pass") or 0, v.get("fail") or 0
        tot = p + f
        out.append({"key": k, "date": _d(k), "pass": p, "fail": f,
                    "total": tot,
                    "rate": round(p * 100.0 / tot, 2) if tot else None})
    return out


def weekly(trend: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Gộp chuỗi ngày thành TUẦN ISO (thứ Hai đầu tuần).

    Tuần ISO chứ không phải "7 ngày tính ngược từ hôm nay": người ở xưởng nói
    "tuần trước" theo lịch, và một cửa sổ trượt thì hai báo cáo lập cách nhau
    một ngày không so được với nhau.

    Nhãn ghi kèm KHOẢNG NGÀY THẬT trong dữ liệu, không phải ngày đầu/cuối tuần
    lịch: tuần đầu của kỳ thường bị cắt, và ghi "17/08–23/08" cho một tuần chỉ
    có số của hai ngày là nói quá phạm vi dữ liệu.
    """
    buckets: Dict[Tuple[int, int], Dict[str, Any]] = {}
    for d in daily(trend):
        if d["date"] is None:
            continue
        iso = d["date"].isocalendar()
        key = (iso[0], iso[1])
        b = buckets.setdefault(key, {
            "year": iso[0], "week": iso[1],
            "monday": d["date"] - timedelta(days=iso[2] - 1),
            "first": d["date"], "last": d["date"],
            "pass": 0, "fail": 0, "days": 0,
        })
        b["first"] = min(b["first"], d["date"])
        b["last"] = max(b["last"], d["date"])
        b["pass"] += d["pass"]
        b["fail"] += d["fail"]
        b["days"] += 1 if d["total"] else 0

    out = []
    for key in sorted(buckets):
        b = buckets[key]
        tot = b["pass"] + b["fail"]
        out.append({
            **b,
            "total": tot,
            "rate": round(b["pass"] * 100.0 / tot, 2) if tot else None,
            # Chuẩn hoá theo NGÀY CÓ CHẠY, không theo 7. Một tuần mới bắt đầu
            # được hai ngày mà chia cho 7 thì máy trông như vừa tụt sản lượng.
            "per_active_day": round(tot / b["days"], 1) if b["days"] else None,
            "label": T("week_short", n=b["week"]),
            "span": (f"{b['first'].strftime('%d/%m')}–{b['last'].strftime('%d/%m')}"
                     if b["first"] != b["last"] else b["first"].strftime("%d/%m")),
            # `partial` = tuần bị KỲ BÁO CÁO cắt (đầu kỳ/cuối kỳ). Với kỳ 7
            # ngày thì mọi tuần đều thế, nên nó không đáng in ra.
            "partial": ((b["last"] - b["first"]).days + 1) < 7,
            # Còn đây là thứ đáng in: trong khoảng đã có số, có bao nhiêu ngày
            # KHÔNG ra sản phẩm. Đó là ngày đứng máy, không phải ngày ngoài kỳ.
            "idle_days": ((b["last"] - b["first"]).days + 1) - b["days"],
        })

    # Delta tính SAU khi đã sắp, và chỉ giữa hai tuần cạnh nhau có số. Bỏ qua
    # bước này thì tuần đầu kỳ mang delta so với hư không.
    prev = None
    for w in out:
        w["delta_points"] = (round(w["rate"] - prev, 2)
                             if w["rate"] is not None and prev is not None else None)
        if w["rate"] is not None:
            prev = w["rate"]
    return out


def machine_view(row: Dict[str, Any], period_days: int) -> Dict[str, Any]:
    """
    Một máy, đủ thứ phụ lục cần — kể cả máy không có dữ liệu.

    Máy lỗi vẫn trả về một view đầy đủ khoá với giá trị None, để lớp vẽ không
    phải rẽ nhánh và không thể vô tình bỏ máy đó ra khỏi báo cáo.
    """
    prod = row.get("production") or {}
    trend = prod.get("trend") or {}
    days = daily(trend)
    weeks = weekly(trend)
    active = [d for d in days if d["total"]]
    total = prod.get("total_products")

    fm = row.get("failure_modes") or {}
    causes = fm.get("by_cause") or []
    top = max(causes, key=lambda c: c.get("share_of_causes_pct") or 0) if causes else None

    worst = min((d for d in active if d["rate"] is not None),
                key=lambda d: d["rate"], default=None)

    return {
        "machine": row.get("machine"),
        "line": row.get("line"),
        "model": row.get("model"),
        "state": row.get("state"),
        "error": row.get("error"),
        "has_data": bool(prod),
        "total": total,
        "pass": prod.get("pass"),
        "fail": prod.get("fail"),
        "rate": prod.get("pass_rate"),
        "per_day": prod.get("per_day"),
        # Hai cách chuẩn hoá, cố ý để cạnh nhau. `per_day` chia cho cả kỳ, nên
        # một máy chạy 2 trong 7 ngày trông như sản lượng thấp; `per_active_day`
        # nói nó chạy nhanh nhưng chạy ít ngày. Chỉ in một trong hai là bỏ mất
        # một nửa câu trả lời cho "vì sao máy này thấp".
        "active_days": len(active),
        "period_days": period_days,
        "per_active_day": (round(total / len(active), 1)
                           if total and active else None),
        "days": days,
        "weeks": weeks,
        "recipes": row.get("recipes") or [],
        "top_cause": top,
        "sample_products": fm.get("sample_products"),
        "sample_covers_all": fm.get("sample_covers_all"),
        "sampling": fm.get("sampling"),
        "template_similarity_avg": fm.get("template_similarity_avg"),
        "worst_day": worst,
    }


# Cần bấy nhiêu máy CÓ SỐ thì trung vị của cột mới đáng in.
MEDIAN_MIN_N = 3


def _median(xs: List[float]) -> Optional[float]:
    """Trung vị, hoặc None khi quá ít máy có số cho cột này.

    Không có ngưỡng thì cột chỉ một máy có số cho ra "trung vị" đúng bằng giá
    trị của máy đó — in ra là mời người đọc so nó với chính nó, và ô đó không
    bao giờ bị tô đậm dù lệch tới đâu. Cùng lỗi đã bỏ biểu đồ một chuỗi ở màn
    hình Line Station.
    """
    xs = sorted(x for x in xs if x is not None)
    if len(xs) < MEDIAN_MIN_N:
        return None
    mid = len(xs) // 2
    return xs[mid] if len(xs) % 2 else (xs[mid - 1] + xs[mid]) / 2.0


def fingerprint_view(fp: Dict[str, Any]) -> Dict[str, Any]:
    """
    Vân tay lỗi kèm TRUNG VỊ từng nguyên nhân và độ lệch của từng máy.

    Trung vị chứ không trung bình: năm máy thì một máy lệch mạnh đủ kéo trung
    bình đi theo nó, và ô lệch mạnh nhất tự làm mờ chính mình.
    """
    causes = fp.get("causes") or []
    by = fp.get("by_machine") or {}
    if not causes or not by:
        return {}
    med = {c: _median([(v.get("by_cause") or {}).get(c) for v in by.values()])
           for c in causes}
    rows = []
    for name, v in by.items():
        cells = []
        for c in causes:
            val = (v.get("by_cause") or {}).get(c)
            m = med.get(c)
            # "Lệch mạnh" = hơn trung vị 12 điểm trở lên. Ngưỡng tuyệt đối chứ
            # không tương đối: trung vị 0,5% thì lệch gấp ba vẫn là 1,5% và
            # không đáng tô đậm.
            cells.append({"cause": c, "value": val,
                          "hot": (val is not None and m is not None
                                  and val - m >= 12)})
        rows.append({"machine": name, "cells": cells,
                     "sample_products": v.get("sample_products"),
                     "covers_all": bool(v.get("sample_covers_all"))})
    return {"causes": causes, "labels": fp.get("cause_labels") or {},
            "median": med, "rows": rows}


# Việc cần làm, tra theo mã nguyên nhân. Nói "no_detection cao" mà không nói
# phải đi xem cái gì thì người đọc vẫn phải hỏi lại một câu nữa.
_CAUSE_ACTION = {
 "vi": {
  "no_detection": "kiểm tra camera/trigger/ánh sáng — detector không thấy vùng in",
  "char_verification": "thấy vùng in nhưng ký tự dưới ngưỡng: xem nét in, mực, tiêu cự",
  "text_verification": "OCR đọc ra chuỗi khác: xem lại chuỗi mong đợi trong recipe",
  "template_verification": "ảnh không khớp template: template có thể đã cũ so với bao bì",
  "product_verification": "không nhận ra sản phẩm: sai recipe đang chạy, hoặc bao bì mới",
 },
 "en": {
  "no_detection": "check camera, trigger and lighting — the detector sees no printed area",
  "char_verification": "print is found but characters fall short: print quality, ink, focus",
  "text_verification": "OCR read a different string: re-check the expected string in the recipe",
  "template_verification": "image does not match the template: it may be out of date against the packaging",
  "product_verification": "product not recognised: wrong recipe loaded, or new packaging",
 },
}


def cause_action(cause: str, fallback_key: str = "f_detail_act") -> str:
    """Việc cần làm cho một mã nguyên nhân. Nói "no_detection cao" mà không nói
    phải đi xem cái gì thì người đọc vẫn phải hỏi lại một câu nữa."""
    return _CAUSE_ACTION.get(LANG, _CAUSE_ACTION["vi"]).get(cause) or T(fallback_key)



def findings(views: List[Dict[str, Any]], fp: Dict[str, Any],
             coverage: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Phát hiện chính, dựng bằng CODE chứ không bằng mô hình.

    Đây là những dòng người đọc sẽ hành động theo, nên chúng phải suy được từ
    đúng những con số in ngay trên trang. Một câu do LLM viết thì hay hơn nhưng
    không kiểm được, và báo cáo là chỗ sai một câu là mất niềm tin cả tập.
    """
    out: List[Dict[str, Any]] = []

    for m in coverage.get("machines_missing") or []:
        out.append({"kind": "bad", "text": T("f_missing", m=m["machine"]),
                    "action": T("f_missing_act")})

    live = [v for v in views if v["has_data"]]

    for v in sorted(live, key=lambda v: v["rate"] or 0):
        if v["rate"] is not None and v["rate"] < RATE_WATCH:
            top = v.get("top_cause") or {}
            out.append({
                "kind": "bad",
                "text": T("f_below", m=v["machine"], rate=fmt_pct(v["rate"]),
                          t=fmt_pct(RATE_WATCH, 0),
                          label=cause_label(top.get("cause"), top.get("label") or "—"),
                          pct=fmt_pct(top.get("share_of_causes_pct") or 0, 0)),
                "action": cause_action(top.get("cause")),
            })
            break

    # Tuần trượt: chỉ tính khi có ít nhất hai tuần CÓ SỐ. So một tuần với hư
    # không là cách tạo ra một phát hiện từ không có gì.
    for v in live:
        ws = [w for w in v["weeks"] if w["rate"] is not None]
        if len(ws) >= 2 and ws[-1]["delta_points"] is not None:
            d = ws[-1]["delta_points"]
            if d <= -3:
                out.append({
                    "kind": "watch",
                    "text": T("f_slip", m=v["machine"], w=ws[-1]["week"],
                              d=fmt_num(abs(d), 2), a=fmt_pct(ws[-2]["rate"]),
                              b=fmt_pct(ws[-1]["rate"])),
                    "action": T("f_slip_act")})
                break

    # Máy chạy ít ngày: nói ra để không ai đọc cột "mỗi ngày" như năng lực máy.
    few = [v for v in live
           if v["active_days"] and v["active_days"] * 2 <= v["period_days"]]
    if few:
        names = ", ".join(T("f_fewdays_item", m=v["machine"], a=v["active_days"],
                            b=v["period_days"]) for v in few)
        out.append({"kind": "info", "text": T("f_fewdays", names=names),
                    "action": T("f_fewdays_act")})

    if fp and fp.get("rows"):
        hot = [(r["machine"], c) for r in fp["rows"] for c in r["cells"] if c["hot"]]
        for name, c in hot[:1]:
            lab = cause_label(c["cause"], (fp.get("labels") or {}).get(c["cause"], c["cause"]))
            out.append({
                "kind": "watch",
                "text": T("f_outlier", m=name, label=lab, pct=fmt_pct(c["value"], 0),
                          med=fmt_pct(fp["median"][c["cause"]], 0)),
                "action": cause_action(c["cause"], "f_outlier_act")})

    if not out:
        out.append({"kind": "good", "text": T("f_none"), "action": T("f_none_act")})
    return out
