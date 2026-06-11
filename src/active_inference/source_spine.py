"""PDF-derived source spine for Namjoshi's *Fundamentals of Active Inference*.

The ledger is intentionally static. It records the table-of-contents surface of
``/Users/4d/Documents/Namjoshi_2025_v5_Fundamentals_of_Active_Inference.pdf``
as inspected for this companion repository: Chapters 1-14 and Appendices A-D.
It is not a claim that every section has a numerically exact worked example.
Validators use it to prevent invented chapters and silent appendix drift.
"""

from __future__ import annotations

from dataclasses import dataclass


SOURCE_PDF_PATH = "/Users/4d/Documents/Namjoshi_2025_v5_Fundamentals_of_Active_Inference.pdf"
SOURCE_PDF_PAGES = 1153
SOURCE_PDF_BUILD_DATE = "2025-03-20"


@dataclass(frozen=True)
class SourceChapter:
    """One manuscript chapter from the inspected PDF table of contents."""

    number: int
    title: str
    sections: tuple[str, ...]


@dataclass(frozen=True)
class SourceAppendix:
    """One manuscript appendix from the inspected PDF table of contents."""

    letter: str
    title: str
    sections: tuple[str, ...]
    executable: bool


CHAPTERS: tuple[SourceChapter, ...] = (
    SourceChapter(1, "The Hypothesis-Testing Brain", ("1.1", "1.2", "1.3", "1.4", "1.5")),
    SourceChapter(2, "Hidden State Estimation", ("2.1", "2.1.1", "2.1.2", "2.1.3", "2.1.4", "2.2", "2.3", "2.4", "2.5", "2.5.1", "2.5.2", "2.6")),
    SourceChapter(3, "Combining Learning and Inference", ("3.1", "3.2", "3.3", "3.4", "3.5", "3.5.1", "3.5.2", "3.6")),
    SourceChapter(4, "Variational Bayesian inference", ("4.1", "4.2", "4.3", "4.4", "4.5", "4.6", "4.7")),
    SourceChapter(5, "Predictive Coding", ("5.1", "5.2", "5.3", "5.4", "5.5")),
    SourceChapter(6, "Generalized filtering for perception", ("6.1", "6.2", "6.3", "6.4", "6.5", "6.6", "6.7")),
    SourceChapter(7, "Active generalized filtering", ("7.1", "7.2", "7.3", "7.4", "7.5", "7.6")),
    SourceChapter(8, "Learning, attention, and hierarchical models in continuous state-spaces", ("8.1", "8.1.1", "8.1.2", "8.2", "8.2.1", "8.2.2", "8.3", "8.4", "8.5", "8.5.1", "8.5.2", "8.5.3", "8.5.4", "8.6")),
    SourceChapter(9, "Active inference in partially observable Markov decision processes", ("9.1", "9.1.1", "9.1.2", "9.2", "9.2.1", "9.2.2", "9.3", "9.4", "9.5", "9.5.1", "9.5.2", "9.6", "9.7")),
    SourceChapter(10, "Learning and other extensions to discrete state-space active inference models", ("10.1", "10.2", "10.3", "10.4", "10.5")),
    SourceChapter(11, "Modifications and extensions to base active inference", ("11.1", "11.1.1", "11.1.2", "11.1.3", "11.1.4", "11.1.5", "11.1.6", "11.1.7", "11.1.8", "11.2", "11.2.1", "11.2.2", "11.2.3", "11.2.4", "11.2.5", "11.2.6", "11.2.7", "11.2.8", "11.2.9", "11.3", "11.4", "11.5", "11.6")),
    SourceChapter(12, "Active inference factor graph models", ("12.1", "12.2", "12.3", "12.4", "12.4.1", "12.5", "12.6", "12.7", "12.8")),
    SourceChapter(13, "Active inference robotics", ("13.1", "13.2", "13.3", "13.4", "13.5")),
    SourceChapter(14, "A free energy principle for the brain", ("14.1", "14.2", "14.3", "14.4")),
)

APPENDICES: tuple[SourceAppendix, ...] = (
    SourceAppendix("A", "History and future of active inference and the free energy principle", ("A.1", "A.1.1", "A.1.2", "A.1.3", "A.1.4", "A.1.5", "A.1.6", "A.1.7", "A.2", "A.2.1", "A.2.2", "A.2.3", "A.2.4", "A.2.5"), False),
    SourceAppendix("B", "Mathematical notation and modeling setup", ("B.1", "B.2", "B.3", "B.4", "B.5", "B.6", "B.7", "B.8", "B.9", "B.10", "B.11", "B.12"), True),
    SourceAppendix("C", "Mathematical fundamentals", ("C.1", "C.1.1", "C.1.2", "C.1.3", "C.1.4", "C.2", "C.2.1", "C.2.2", "C.2.2.1", "C.2.2.2", "C.2.2.3", "C.2.2.4", "C.2.2.5", "C.2.3", "C.2.4", "C.2.5", "C.2.6", "C.3", "C.3.1", "C.3.2", "C.4", "C.4.1", "C.4.2", "C.4.3", "C.4.4", "C.5", "C.5.1", "C.5.2", "C.5.3", "C.5.4", "C.6", "C.6.1", "C.6.2", "C.7", "C.7.1", "C.7.2", "C.7.3", "C.8", "C.8.1", "C.8.2", "C.8.3", "C.9", "C.10", "C.10.1", "C.10.2", "C.10.3", "C.10.4", "C.10.5", "C.10.6", "C.10.7", "C.11", "C.11.1", "C.11.2", "C.11.3", "C.11.4", "C.12", "C.13", "C.13.1", "C.13.2"), True),
    SourceAppendix("D", "Forms of free energy", ("D.1", "D.2", "D.3", "D.3.1", "D.3.2", "D.3.3", "D.3.4", "D.4"), True),
)


def chapter_numbers() -> tuple[int, ...]:
    """Return the chapter numbers present in the inspected PDF."""
    return tuple(chapter.number for chapter in CHAPTERS)


def appendix_letters() -> tuple[str, ...]:
    """Return appendix letters present in the inspected PDF."""
    return tuple(appendix.letter for appendix in APPENDICES)


def chapter_section_ids() -> tuple[str, ...]:
    """Return all chapter section identifiers from the inspected PDF."""
    return tuple(section for chapter in CHAPTERS for section in chapter.sections)


def appendix_section_ids() -> tuple[str, ...]:
    """Return all appendix section identifiers from the inspected PDF."""
    return tuple(section for appendix in APPENDICES for section in appendix.sections)


def all_section_ids() -> tuple[str, ...]:
    """Return every chapter and appendix section identifier in source order."""
    return chapter_section_ids() + appendix_section_ids()


def has_chapter(number: int) -> bool:
    """Return whether ``number`` exists in the inspected PDF source spine."""
    return int(number) in chapter_numbers()


def has_section(identifier: str) -> bool:
    """Return whether ``identifier`` exists in the inspected PDF source spine."""
    return identifier in all_section_ids()


__all__ = [
    "APPENDICES",
    "CHAPTERS",
    "SOURCE_PDF_BUILD_DATE",
    "SOURCE_PDF_PAGES",
    "SOURCE_PDF_PATH",
    "SourceAppendix",
    "SourceChapter",
    "all_section_ids",
    "appendix_letters",
    "appendix_section_ids",
    "chapter_numbers",
    "chapter_section_ids",
    "has_chapter",
    "has_section",
]
