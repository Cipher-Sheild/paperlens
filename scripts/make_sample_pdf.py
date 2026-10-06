"""Generate sample_data/sample_paper.pdf (needs reportlab: pip install reportlab)."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from paperlens import sample as S

FILLER = ("This section describes material that is not needed for the summary. " * 12).strip()


def build(path):
    st = getSampleStyleSheet()
    H, B = st["Heading2"], st["BodyText"]
    story = [Paragraph("Summarizing Research Papers with BERT", st["Title"]), Spacer(1, 10),
             Paragraph("Abstract", H), Paragraph(S.ABSTRACT, B),
             Paragraph("1 Introduction", H), Paragraph(S.INTRO_1, B), Paragraph(S.INTRO_2, B),
             Paragraph("1.1 Contributions", st["Heading3"]), Paragraph(S.INTRO_3, B),
             Paragraph("2 Related Work", H), Paragraph(FILLER, B),
             Paragraph("3 Method", H), Paragraph(FILLER, B),
             Paragraph("4 Experiments", H), Paragraph(FILLER, B),
             Paragraph("5 Conclusion", H), Paragraph(S.CONCLUSION, B),
             Paragraph("References", H), Paragraph("[1] Devlin et al. BERT. 2019.", B)]
    SimpleDocTemplate(path, pagesize=A4).build(story)


if __name__ == "__main__":
    out = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "sample_data", "sample_paper.pdf")
    build(out)
    print("written", out)
