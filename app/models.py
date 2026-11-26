from pydantic import BaseModel, Field

SENIORITY_RANK = {"intern": 0, "junior": 1, "mid": 2, "senior": 3}


class Job(BaseModel):
    id: str
    title: str
    company: str
    location: str
    remote: bool
    seniority: str
    salary_min: int | None = None  # annual, thousands of USD
    salary_max: int | None = None
    skills: list[str]
    description: str

    def text(self) -> str:
        """Text used for retrieval and for dealbreaker checks."""
        return f"{self.title}. {self.company}. {' '.join(self.skills)}. {self.description}"


class Profile(BaseModel):
    """Structured candidate profile extracted from free text."""

    skills: list[str] = Field(default_factory=list)
    roles: list[str] = Field(default_factory=list)
    seniority: str | None = None  # intern | junior | mid | senior
    locations: list[str] = Field(default_factory=list)
    remote_only: bool = False
    min_salary: int | None = None  # annual, thousands of USD
    dealbreakers: list[str] = Field(default_factory=list)  # terms that must NOT appear in a posting
    summary: str = ""

    def query_text(self) -> str:
        return " ".join([*self.roles, *self.skills, self.summary]).strip()
