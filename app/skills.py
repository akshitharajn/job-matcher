"""Skill vocabulary and extraction, used to give live postings the same `skills` field as sample ones."""
from .text import contains_term

# Feed tags that describe the posting, not a skill. Counting them would make "remote" or "engineering"
# look like a skill that matches almost every job.
GENERIC_TAGS = frozenset("""
remote engineering engineer developer software tech technology full-time part-time contract freelance senior
junior lead manager english europe usa uk worldwide anywhere marketing sales support design product operations
management consulting finance education healthcare business internship intern graduate startup digital
""".split())

SKILL_VOCAB = [
    # languages (single-letter / common-word names like "c", "r", "go" are left out: they match ordinary prose)
    "python", "java", "javascript", "typescript", "c++", "c#", "golang", "rust", "kotlin", "swift", "php",
    "ruby", "scala", "sql", "matlab", "dart", "bash", "embedded c",
    # web / backend
    "react", "angular", "vue", "node", "django", "flask", "fastapi", "spring", "rails", "graphql", "rest",
    "html", "css", "next.js", "express",
    # data / ml / ai
    "pandas", "numpy", "scikit-learn", "pytorch", "tensorflow", "keras", "spark", "airflow", "dbt", "tableau",
    "power bi", "excel", "nlp", "computer vision", "machine learning", "deep learning", "llm", "rag",
    "embeddings", "prompt engineering", "transformers", "langchain", "opencv", "mlops", "recommender systems",
    "data science", "data engineering", "statistics", "a/b testing", "evaluation",
    # infra
    "docker", "kubernetes", "aws", "azure", "gcp", "terraform", "ci/cd", "linux", "git", "kafka",
    "postgresql", "mysql", "mongodb", "redis", "elasticsearch", "firebase", "microservices",
    # mobile / game
    "flutter", "react native", "android", "ios", "unity",
    # testing / security
    "selenium", "pytest", "cypress", "siem", "networking", "penetration testing",
    # electrical / embedded
    "embedded", "firmware", "pcb", "verilog", "vhdl", "fpga", "rtos", "simulink", "labview", "plc",
    "analog", "power electronics", "vlsi", "iot", "arduino", "raspberry pi", "circuit design", "signal processing",
]


def extract_skills(tags: list | None, text: str, cap: int = 12) -> list[str]:
    """Feed tags that are real skills first, then vocabulary terms found in the title/description.
    Free-form tags are ignored: some feeds use them for job categories ("Data Scientist"), which would
    show up as bogus skill gaps."""
    out: list[str] = []
    for tag in tags or []:
        t = str(tag).strip().lower()
        if t in SKILL_SET and t not in GENERIC_TAGS and t not in out:
            out.append(t)
    for term in SKILL_VOCAB:
        if term not in out and contains_term(text, term):
            out.append(term)
    return out[:cap]


SKILL_SET = frozenset(SKILL_VOCAB)
