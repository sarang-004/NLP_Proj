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

    resume_chunks = chunk_text(
        preprocess(resume)
    )

    jd_chunks = chunk_text(
        preprocess(job_description)
    )

    if not resume_chunks or not jd_chunks:
        return 0.0

    resume_embeddings = model.encode(
        resume_chunks,
        normalize_embeddings=True
    )

    jd_embeddings = model.encode(
        jd_chunks,
        normalize_embeddings=True
    )

    similarity_matrix = cosine_similarity(
        resume_embeddings,
        jd_embeddings
    )

    # Best resume match for each JD chunk
    best_matches = np.max(
        similarity_matrix,
        axis=0
    )

    semantic_score = np.mean(
        best_matches
    )

    # Keep score between 0 and 1
    semantic_score = np.clip(
        semantic_score,
        0,
        1
    )

    return float(semantic_score)


# ============================================================
# SKILLS SCORE
# ============================================================

def calculate_skill_score(
    resume,
    job_description
):

    resume_skills = extract_skills(
        resume
    )

    required_skills = extract_skills(
        job_description
    )

    if not required_skills:
        return 0.0

    matched_skills = (
        resume_skills &
        required_skills
    )

    score = (
        len(matched_skills)
        /
        len(required_skills)
    )

    return float(score)


# ============================================================
# COMPLETE SCREENING
# ============================================================

def screen_resume(
    resume,
    job_description
):

    semantic_score = calculate_semantic_score(
        resume,
        job_description
    )

    skills_score = calculate_skill_score(
        resume,
        job_description
    )

    final_score = (
        SEMANTIC_WEIGHT * semantic_score
        +
        SKILLS_WEIGHT * skills_score
    )

    return {
        "semantic_score": round(
            semantic_score, 2
        ),

        "skills_score": round(
            skills_score, 2
        ),

        "final_score": round(
            final_score, 2
        )
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