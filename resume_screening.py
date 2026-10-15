import re
import os
import numpy as np

from pypdf import PdfReader
from docx import Document

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

SEMANTIC_WEIGHT = 0.70
SKILLS_WEIGHT = 0.30

MAX_WORDS_PER_CHUNK = 180


# ============================================================
# LOAD MINILM MODEL
# ============================================================

print("Loading MiniLM model...")
model = SentenceTransformer(MODEL_NAME)
print("Model loaded successfully!\n")


# ============================================================
# READ PDF RESUME
# ============================================================

def read_pdf(file_path):

    reader = PdfReader(file_path)

    text = ""

    for page in reader.pages:
        page_text = page.extract_text()

        if page_text:
            text += page_text + "\n"

    return text


# ============================================================
# READ DOCX RESUME
# ============================================================

def read_docx(file_path):

    document = Document(file_path)

    text = ""

    for paragraph in document.paragraphs:
        text += paragraph.text + "\n"

    return text


# ============================================================
# READ TXT RESUME
# ============================================================

def read_txt(file_path):

    with open(file_path, "r", encoding="utf-8") as file:
        return file.read()


# ============================================================
# EXTRACT RESUME TEXT
# ============================================================
def extract_resume_text(file_path):

    # Remove quotes if user copied the Windows path
    file_path = file_path.strip().strip('"').strip("'")

    # Convert to absolute path
    file_path = os.path.abspath(file_path)

    print(f"\nLooking for file:")
    print(file_path)

    if not os.path.isfile(file_path):
        raise FileNotFoundError(
            f"File does not exist:\n{file_path}"
        )

    extension = os.path.splitext(file_path)[1].lower()

    if extension == ".pdf":
        return read_pdf(file_path)

    elif extension == ".docx":
        return read_docx(file_path)

    elif extension == ".txt":
        return read_txt(file_path)

    else:
        raise ValueError(
            "Unsupported file type. Use PDF, DOCX, or TXT."
        )

# ============================================================
# PREPROCESSING
# ============================================================

def preprocess(text):

    text = text.lower()

    text = re.sub(r"\s+", " ", text)

    text = re.sub(
        r"[^\w\s+#./-]",
        " ",
        text
    )

    return text.strip()


# ============================================================
# CHUNKING
# ============================================================

def chunk_text(text, max_words=MAX_WORDS_PER_CHUNK):

    words = text.split()

    chunks = []

    for i in range(0, len(words), max_words):

        chunk = " ".join(
            words[i:i + max_words]
        )

        if chunk.strip():
            chunks.append(chunk)

    return chunks


# ============================================================
# SKILL DATABASE
# ============================================================

SKILLS = {
    # Programming
    "python",
    "java",
    "c",
    "c++",
    "c#",
    "javascript",
    "typescript",
    "go",
    "rust",
    "php",

    # Web
    "html",
    "css",
    "react",
    "react.js",
    "angular",
    "vue",
    "node.js",
    "node",
    "express",
    "next.js",
    "nextjs",

    # Databases
    "mysql",
    "postgresql",
    "mongodb",
    "oracle",
    "sqlite",
    "redis",
    "firebase",
    "sql",

    # Data Science / ML
    "machine learning",
    "deep learning",
    "artificial intelligence",
    "data science",
    "pandas",
    "numpy",
    "scikit-learn",
    "tensorflow",
    "pytorch",
    "keras",

    # NLP / AI
    "nlp",
    "natural language processing",
    "transformers",
    "bert",
    "llm",
    "generative ai",

    # Cloud / DevOps
    "aws",
    "azure",
    "gcp",
    "docker",
    "kubernetes",
    "git",
    "github",
    "linux",

    # Tools
    "power bi",
    "tableau",
    "excel",
    "flask",
    "django",
    "spring",
    "spring boot",

    # APIs
    "rest api",
    "api"
}


# ============================================================
# EXTRACT SKILLS
# ============================================================

def extract_skills(text):

    text = preprocess(text)

    found_skills = set()

    for skill in SKILLS:

        pattern = (
            r"(?<!\w)"
            + re.escape(skill.lower())
            + r"(?!\w)"
        )

        if re.search(pattern, text):

            found_skills.add(skill)

    return found_skills


# ============================================================
# SEMANTIC SCORE
# ============================================================

def calculate_semantic_score(
    resume,
    job_description
):
    from screening.upload_workflow import evaluate_uploads, make_upload_job
    job = make_upload_job({"job_title": "Resume screening", "job_description": job_description})
    result = evaluate_uploads([("Candidate", resume)], job, model)[0]
    return float(result["component_scores"]["semantic"])


# ============================================================
# SKILLS SCORE
# ============================================================

def calculate_skill_score(
    resume,
    job_description
):
    from screening.upload_workflow import evaluate_uploads, make_upload_job
    job = make_upload_job({"job_title": "Resume screening", "job_description": job_description})
    result = evaluate_uploads([("Candidate", resume)], job, model)[0]
    value = result["component_scores"]["skills"]
    return float(value) if value is not None else None


# ============================================================
# COMPLETE SCREENING
# ============================================================

def screen_resume(
    resume,
    job_description
):
    from screening.upload_workflow import evaluate_uploads, make_upload_job
    job = make_upload_job({"job_title": "Resume screening", "job_description": job_description})
    result = evaluate_uploads([("Candidate", resume)], job, model)[0]
    return {
        "score_methodology_version": result["score_methodology_version"],
        "semantic_score": result["component_scores"]["semantic"],
        "skills_score": result["component_scores"]["skills"],
        "experience_score": result["component_scores"]["experience"],
        "education_course_score": result["component_scores"]["education_course"],
        "projects_score": result["component_scores"]["projects"],
        "effective_weights": result["effective_weights"],
        "weighted_contributions": result["weighted_contributions"],
        "final_score": result["final_score"],
    }


# ============================================================
# MAIN PROGRAM
# ============================================================

if __name__ == "__main__":

    print("=" * 60)
    print("        RESUME SCREENING SYSTEM")
    print("=" * 60)

    # --------------------------------------------------------
    # Resume input
    # --------------------------------------------------------

    resume_path = input(
        "\nEnter resume file path (PDF/DOCX/TXT): "
    )

    resume_path = resume_path.strip().strip('"').strip("'")
    try:

        resume_text = extract_resume_text(
            resume_path
        )

        if not resume_text.strip():
            print("\nERROR: Could not extract text from resume.")
            exit()

        print("\nResume loaded successfully.")

    except Exception as e:

        print(f"\nERROR: {e}")
        exit()

    # --------------------------------------------------------
    # Job description input
    # --------------------------------------------------------

    print("\nEnter Job Description.")
    print("Type END on a new line when finished.\n")

    jd_lines = []

    while True:

        line = input()

        if line.strip().upper() == "END":
            break

        jd_lines.append(line)

    job_description = "\n".join(
        jd_lines
    )

    if not job_description.strip():

        print("\nERROR: Job description is empty.")
        exit()

    # --------------------------------------------------------
    # Run screening
    # --------------------------------------------------------

    print("\nAnalyzing resume...")
    print("Please wait...\n")

    result = screen_resume(
        resume_text,
        job_description
    )

    # --------------------------------------------------------
    # Output
    # --------------------------------------------------------

    print("=" * 60)
    print("              SCREENING RESULT")
    print("=" * 60)

    print(
        f"\nSemantic Score : "
        f"{result['semantic_score']:.2f}"
    )

    print(
        f"Skills Score   : "
        f"{result['skills_score']:.2f}"
    )

    print(
        f"Final Score    : "
        f"{result['final_score']:.2f}"
    )

    print("\n" + "=" * 60)
