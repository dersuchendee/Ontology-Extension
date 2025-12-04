from __future__ import annotations
import argparse
import asyncio
import csv
import os
import re
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple, Literal
import tempfile

import faiss  # type: ignore
import numpy as np
import pandas as pd
from pydantic import BaseModel, Field
from rdflib import BNode, Graph, Literal, OWL, RDF, RDFS, URIRef
from rich import print
from tiktoken import get_encoding
from dotenv import load_dotenv
from llm_service import TokenCostTracker, get_llm_service
from embedder import get_embedder
try:
    from owlready2 import World, Restriction  # type: ignore
except ImportError as exc:  # pragma: no cover - runtime guard
    raise ImportError("owlready2 is required for OntoDeside post-processing. Please install it (pip install owlready2).") from exc
# ==========================
# Configuration & Constants
# ==========================

load_dotenv()

LLM_SERVICE = os.getenv("LLM_SERVICE", "openai")
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o")
LLM_TEMPERATURE = os.getenv("LLM_TEMPERATURE", 0.7)

EMBEDDER = os.getenv("EMBEDDER", "openai")
EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-small")
VECTOR_DIM = os.getenv("VECTOR_DIM", 1536)

ENC = get_encoding("cl100k_base")

MAX_GENERATION_ATTEMPTS = int(os.getenv("ONTOEXTEND_MAX_ATTEMPTS", "3"))

STANDARD_PREFIXES: Dict[str, str] = {
    "rdf": "http://www.w3.org/1999/02/22-rdf-syntax-ns#",
    "rdfs": "http://www.w3.org/2000/01/rdf-schema#",
    "owl": "http://www.w3.org/2002/07/owl#",
    "xsd": "http://www.w3.org/2001/XMLSchema#",
}

REQUIRED_PREFIXES: List[Tuple[str, str]] = [
    ("rdf", "http://www.w3.org/1999/02/22-rdf-syntax-ns#"),
    ("rdfs", "http://www.w3.org/2000/01/rdf-schema#"),
    ("owl", "http://www.w3.org/2002/07/owl#"),
    ("xsd", "http://www.w3.org/2001/XMLSchema#"),
]

SYS_PROMPT = (
    "You are an ontology engineer. Reuse elements from the provided core ontology when possible. "
    "If you must create new elements, append '# GENERATED' as a comment. "
    "Return only Turtle/TTL syntax inside one markdown code block."
)

USER_TMPL = """
You are a helpful assistant designed to generate ontologies. You receive a COMPETENCY QUESTION (CQ) and optionally an ONTOLOGY STORY (OS). \n

Based on CQ, which is a requirement for the ontology, and OS, which tells you what the context of the ontology is, your task is generating one ontology (O). The goal is to generate O that models the CQ properly. This means there is a way to write a SPARQL query to extract the answer to this CQ in O.  \n

Reuse the relevant ontology elements provided in the RELEVANT ONTOLOGY ELEMENTS section below whenever possible. These elements come from multiple reference ontologies. Only create new elements if absolutely necessary. In any case, add labels and comments. \n

Use the following prefixes:

{prefix_block}

 

Don't put any A-Box (instances) in the ontology and just generate the OWL file using Turtle syntax. Include the entities mentioned in the CQ. Remember to use restrictions when the CQ implies it. The output should be self-contained without any errors. Outside of the code box don't put any comment.\n
 

INSTRUCTIONS:

1. Analyze the CQ to understand what concepts and relationships are needed

2. Map the required concepts to classes/properties from the RELEVANT ONTOLOGY ELEMENTS above

3. Prefer elements with higher semantic similarity to the CQ concepts

4. Output only the TBox ontology in Turtle syntax (no instances/ABox)

5. Include restrictions (e.g., cardinality, value restrictions) when the CQ implies them. Express constraints using OWL restriction constructs only (no SHACL).

6. Reuse every provided IRI exactly as defined; never recreate the class/property declarations. Reference them and add new restrictions/axioms that point to them.

7. When modelling "A or B" scenarios, reuse existing classes via inline owl:unionOf or restrictions. Do not mint helper classes that simply restate existing types unless the CQ explicitly demands a new concept.

8. When a CQ describes that an existing class must use a property (e.g., a process has a catalyst as input), add the necessary owl:Restriction to that class instead of redefining the class hierarchy.

9. Ensure the output is syntactically correct and self-contained

 

COMPETENCY QUESTION: "{cq}"

 

RELEVANT ONTOLOGY ELEMENTS (from {num_ontologies} reference ontologies):

{relevant_elements}

 

NOTES:

- Do not use external ontologies. Only build on the ontologies provided in RELEVANT ONTOLOGY ELEMENTS .

- Reuse the namespaces (ontology prefix) used in the RELEVANT ONTOLOGY ELEMENTS whenever possible. Only if none are provided, use the ":" namespace .

- Add rdfs:label and rdfs:comment annotations for every class and property you introduce.

- Do not duplicate axioms for classes/properties that already exist in the reference snippets—reference them instead.

- Do not create any new owl:Ontology statements .
"""

# ==========================
# Paths & IO
# ==========================

@dataclass
class Paths:
    cache_dir: Path = Path.home() / ".core_ontology_rag_cache"

    def __post_init__(self) -> None:
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        (self.cache_dir / "individual_ontologies").mkdir(parents=True, exist_ok=True)
        (self.cache_dir / "retrieved_elements").mkdir(parents=True, exist_ok=True)

    @property
    def results_csv(self) -> Path:
        return self.cache_dir / "generation_runs.csv"

    @property
    def elements_csv(self) -> Path:
        return self.cache_dir / "ontology_elements.csv"

    @property
    def processing_log_csv(self) -> Path:
        return self.cache_dir / "cq_processing_log.csv"

    @property
    def token_cost_csv(self) -> Path:
        return self.cache_dir / "token_cost_log.csv"

    @property
    def combined_ttl(self) -> Path:
        return self.cache_dir / "combined_ontology.ttl"

    @property
    def individual_dir(self) -> Path:
        return self.cache_dir / "individual_ontologies"

    @property
    def retrieved_dir(self) -> Path:
        return self.cache_dir / "retrieved_elements"


class RunLogger:
    def __init__(self, paths: Paths, tracker: TokenCostTracker) -> None:
        self.paths = paths
        self.tracker = tracker

    def log_elements(self, elements: List["OntologyElement"]) -> None:
        self.paths.elements_csv.parent.mkdir(parents=True, exist_ok=True)
        with self.paths.elements_csv.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow([
                "timestamp", "uri", "element_type", "source_ontology", "label", "comment",
                "domain", "range", "super_classes", "sub_classes",
            ])
            ts = datetime.now().isoformat()
            for e in elements:
                w.writerow([
                    ts,
                    e.uri,
                    e.element_type,
                    Path(e.source_ontology).name,
                    e.label or "",
                    e.comment or "",
                    ";".join(e.domain or []),
                    ";".join(e.range or []),
                    ";".join(e.super_classes or []),
                    ";".join(e.sub_classes or []),
                ])

    def log_processing_result(
        self,
        cq_index: int,
        cq: str,
        success: bool,
        elements_used: int,
        source_ontologies: Dict[str, int],
        processing_time: float,
        error: Optional[str] = None,
    ) -> None:
        self.paths.processing_log_csv.parent.mkdir(parents=True, exist_ok=True)
        newfile = not self.paths.processing_log_csv.exists()
        with self.paths.processing_log_csv.open("a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            if newfile:
                w.writerow([
                    "timestamp", "cq_index", "cq", "success", "elements_used",
                    "source_ontologies_used", "processing_time_seconds", "error",
                ])
            sources_str = ";".join([f"{Path(k).stem}:{v}" for k, v in source_ontologies.items()])
            w.writerow([
                datetime.now().isoformat(),
                cq_index,
                cq[:200] + "..." if len(cq) > 200 else cq,
                success,
                elements_used,
                sources_str,
                f"{processing_time:.2f}",
                error or "",
            ])

    def log_final_results(
        self,
        total_cqs: int,
        successful: int,
        failed: int,
        total_time: float,
        output_file: str,
    ) -> None:
        self.paths.results_csv.parent.mkdir(parents=True, exist_ok=True)
        with self.paths.results_csv.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow([
                "timestamp", "total_cqs", "successful", "failed", "success_rate_percent",
                "total_time_minutes", "avg_time_per_cq_seconds", "output_file",
            ])
            success_rate = (successful / total_cqs * 100) if total_cqs > 0 else 0
            avg_time = (total_time / total_cqs) if total_cqs > 0 else 0
            w.writerow([
                datetime.now().isoformat(),
                total_cqs,
                successful,
                failed,
                f"{success_rate:.1f}",
                f"{total_time / 60:.2f}",
                f"{avg_time:.2f}",
                output_file,
            ])

    def log_cost_row(self, cq_index: int, cq: str, deltas: Dict[str, Dict[str, int]]) -> None:
        self.paths.token_cost_csv.parent.mkdir(parents=True, exist_ok=True)
        newfile = not self.paths.token_cost_csv.exists()
        with self.paths.token_cost_csv.open("a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            if newfile:
                w.writerow([
                    "timestamp", "cq_index", "cq", "model",
                    "prompt_tokens", "completion_tokens", "embedding_tokens", "cost_usd",
                ])
            for model, counts in deltas.items():
                cost = self.tracker.model_cost_usd(model, counts)
                w.writerow([
                    datetime.now().isoformat(),
                    cq_index,
                    cq[:200] + ("..." if len(cq) > 200 else ""),
                    model,
                    counts.get("prompt", 0),
                    counts.get("completion", 0),
                    counts.get("embedding", 0),
                    f"{cost:.6f}",
                ])


# ==========================
# Ontology Elements
# ==========================

class OntologyElement(BaseModel):
    uri: str
    element_type: str  # 'class', 'object_property', 'data_property', 'annotation_property'
    source_ontology: str
    label: Optional[str] = None
    comment: Optional[str] = None
    domain: Optional[List[str]] = Field(default_factory=list)
    range: Optional[List[str]] = Field(default_factory=list)
    super_classes: Optional[List[str]] = Field(default_factory=list)
    sub_classes: Optional[List[str]] = Field(default_factory=list)
    additional_info: Optional[List[str]] = Field(default_factory=list)
    snippet: Optional[str] = None

    def get_searchable_text(self) -> str:
        parts: List[str] = []
        local_name = self.uri.split('#')[-1].split('/')[-1]
        parts.append(local_name)
        if self.label:
            parts.append(self.label)
        if self.comment:
            parts.append(self.comment)
        parts.append(f"Type: {self.element_type.replace('_', ' ')}")
        if self.domain:
            domain_names = [d.split('#')[-1].split('/')[-1] for d in self.domain]
            parts.append(f"Domain: {', '.join(domain_names)}")
        if self.range:
            range_names = [r.split('#')[-1].split('/')[-1] for r in self.range]
            parts.append(f"Range: {', '.join(range_names)}")
        if self.additional_info:
            parts.extend(self.additional_info)
        return " | ".join(parts)


@dataclass
class FragmentReviewResult:
    accepted: bool
    fragment: str
    issues: List[str]
    feedback_hint: Optional[str]
    details: Dict[str, Any]


STRUCTURAL_PREDICATES: Set[URIRef] = {
    RDF.type,
    RDFS.domain,
    RDFS.range,
    RDFS.subClassOf,
    RDFS.subPropertyOf,
    OWL.equivalentClass,
    OWL.equivalentProperty,
    OWL.disjointWith,
}

MANDATORY_KEYWORDS = [
    "must",
    "requires",
    "require",
    "mandatory",
    "obligatory",
    "always",
    "shall",
    "needs to",
]

CATALYST_KEYWORDS = ["catalyst", "catalysts"]


class FragmentReviewer:
    def __init__(self) -> None:
        pass

    def review(self, fragment: str, cq: str, protected: List[OntologyElement]) -> FragmentReviewResult:
        details: Dict[str, Any] = {}
        try:
            graph = Graph()
            graph.parse(data=fragment, format="turtle")
        except Exception as e:
            issue_msg = f"Turtle parse failed: {e}"
            return FragmentReviewResult(False, fragment, [issue_msg], "Ensure the fragment serializes as valid Turtle before returning it.", details)

        protected_uris = {URIRef(e.uri) for e in protected if e.uri}
        redeclared = self._find_redeclarations(graph, protected_uris)

        issues: List[str] = []
        if redeclared:
            issues.append("redefinition")
            details["redefinition"] = redeclared

        if self._has_unwanted_catalyst_requirement(fragment, cq):
            issues.append("catalyst_optional")

        accepted = not issues
        feedback = self._feedback_for(issues, details)
        return FragmentReviewResult(accepted, fragment, issues, feedback, details)

    @staticmethod
    def _find_redeclarations(graph: Graph, protected_uris: Set[URIRef]) -> Dict[str, List[str]]:
        offenders: Dict[str, List[str]] = {}
        if not protected_uris:
            return offenders
        for uri in protected_uris:
            hits: List[str] = []
            for predicate in STRUCTURAL_PREDICATES:
                if (uri, predicate, None) in graph:
                    hits.append(FragmentReviewer._predicate_name(predicate))
            if hits:
                offenders[str(uri)] = hits
        return offenders

    @staticmethod
    def _predicate_name(predicate: URIRef) -> str:
        local = predicate.split('#')[-1].split('/')[-1]
        return local or str(predicate)

    def _has_unwanted_catalyst_requirement(self, fragment: str, cq: str) -> bool:
        cq_lower = cq.lower()
        if not any(keyword in cq_lower for keyword in CATALYST_KEYWORDS):
            return False
        if self._cq_requires_catalyst(cq_lower):
            return False
        try:
            return self._fragment_requires_catalyst(fragment)
        except Exception:
            return False

    @staticmethod
    def _cq_requires_catalyst(cq_lower: str) -> bool:
        return any(keyword in cq_lower for keyword in MANDATORY_KEYWORDS)

    def _fragment_requires_catalyst(self, fragment: str) -> bool:
        with tempfile.NamedTemporaryFile("w", suffix=".ttl", delete=True) as tmp:
            tmp.write(fragment)
            tmp.flush()
            world = World()
            onto = world.get_ontology(tmp.name).load()
            for cls in onto.classes():
                for constraint in cls.is_a:
                    if not isinstance(constraint, Restriction):
                        continue
                    prop_iri = getattr(getattr(constraint, "property", None), "iri", "")
                    if not self._is_catalyst_property(prop_iri):
                        continue
                    if getattr(constraint, "type", None) == "some" or getattr(constraint, "cardinality", None):
                        return True
                    min_card = getattr(constraint, "min", None)
                    if isinstance(min_card, int) and min_card >= 1:
                        return True
            return False

    @staticmethod
    def _is_catalyst_property(prop_iri: str) -> bool:
        return any(keyword in prop_iri.lower() for keyword in CATALYST_KEYWORDS)

    @staticmethod
    def _feedback_for(issues: List[str], details: Dict[str, Any]) -> Optional[str]:
        if not issues:
            return None
        hints: List[str] = []
        if "redefinition" in issues:
            offenders = details.get("redefinition", {})
            offender_list = ", ".join(offenders.keys()) if offenders else "the provided IRIs"
            hints.append(
                f"Do not redefine {offender_list}; reference them without restating domain/range or hierarchy."
            )
        if "catalyst_optional" in issues:
            hints.append("Treat catalyst participation as optional unless the CQ explicitly states it is mandatory.")
        return " ".join(hints)


# ==========================
# Prefix Management
# ==========================

class PrefixManager:
    def __init__(self) -> None:
        pass

    @staticmethod
    def sanitize_prefix_name(raw: Optional[str]) -> str:
        s = (raw or "").strip()
        s = re.sub(r"[^A-Za-z0-9_\-]", "_", s)
        if not s or not re.match(r"^[A-Za-z_]", s):
            s = f"ns_{s}" if s else "ns"
        s = re.sub(r"_+", "_", s)
        return s

    def collect_prefixes_from_files(self, ontology_files: List[str]) -> Tuple[Optional[str], Dict[str, str]]:
        extra_prefixes: Dict[str, str] = {}
        used: set[str] = set(STANDARD_PREFIXES.keys())
        default_ns: Optional[str] = None
        for f in ontology_files:
            g = Graph()
            try:
                g.parse(f)
            except Exception:
                continue
            for pref, ns in g.namespaces():
                ns = str(ns)
                if not ns:
                    continue
                pref_str = "" if pref is None else str(pref)
                if pref_str.lower() == "xml":
                    continue
                if pref_str == "":
                    if not default_ns:
                        default_ns = ns
                    continue
                if pref_str in STANDARD_PREFIXES:
                    continue
                p = self.sanitize_prefix_name(pref_str)
                while p in used:
                    m = re.search(r"(\d+)$", p)
                    p = re.sub(r"\d+$", str(int(m.group(1)) + 1), p) if m else f"{p}2"
                extra_prefixes[p] = ns
                used.add(p)
        return default_ns, extra_prefixes

    def build_prefix_block(self, ontology_files: List[str]) -> str:
        default_ns, extra_prefixes = self.collect_prefixes_from_files(ontology_files)
        lines: List[str] = []
        used: Set[str] = set()
        if default_ns:
            lines.append(f"@prefix : <{default_ns}> .")
            used.add("")
        for pref, uri in REQUIRED_PREFIXES:
            if pref in used:
                continue
            lines.append(f"@prefix {pref}: <{uri}> .")
            used.add(pref)
        for pref in sorted(extra_prefixes.keys(), key=str.lower):
            if pref in used:
                continue
            lines.append(f"@prefix {pref}: <{extra_prefixes[pref]}> .")
            used.add(pref)
        return "\n".join(lines) + "\n"


# ==========================
# Extraction
# ==========================

class OntologyExtractor:
    def __init__(self, logger: RunLogger) -> None:
        self.logger = logger

    @staticmethod
    def _collect_snippet(g: Graph, subject: URIRef, depth: int = 1) -> str:
        snippet = Graph()
        for pref, ns in g.namespaces():
            if pref is None:
                snippet.bind('', ns)
            else:
                snippet.bind(pref, ns)

        visited_bnodes: Set[BNode] = set()

        def add_triples(node: Any, remaining: int) -> None:
            for pred, obj in g.predicate_objects(node):
                snippet.add((node, pred, obj))
                if remaining > 0 and isinstance(obj, BNode) and obj not in visited_bnodes:
                    visited_bnodes.add(obj)
                    add_triples(obj, remaining - 1)

        add_triples(subject, depth)
        try:
            return snippet.serialize(format="turtle").strip()
        except Exception:
            return ""

    def extract_from_files(self, ontology_files: List[str], *, log: bool = True) -> List[OntologyElement]:
        print(f"[blue]Loading {len(ontology_files)} reference ontologies...[/blue]")
        all_elements: List[OntologyElement] = []
        for ontology_file in ontology_files:
            p = Path(ontology_file)
            if not p.exists():
                print(f"[red]Ontology file not found: {ontology_file}[/red]")
                continue
            print(f"[cyan]  Processing: {p.name}[/cyan]")
            try:
                elements = self._extract_single(str(p))
                all_elements.extend(elements)
                print(f"[green] Extracted {len(elements)} elements[/green]")
            except Exception as e:
                print(f"[red]Error processing {ontology_file}: {e}[/red]")
                continue
        print(f"[green]✅ Total extracted {len(all_elements)} elements from {len(ontology_files)} ontologies[/green]")
        if log:
            self.logger.log_elements(all_elements)
        return all_elements

    def _extract_single(self, ontology_file: str) -> List[OntologyElement]:
        g = Graph()
        g.parse(ontology_file)
        elements: List[OntologyElement] = []
        # classes
        for cls in list(g.subjects(RDF.type, OWL.Class)):
            if isinstance(cls, URIRef):
                el = self._extract_class_info(g, cls, ontology_file)
                if el:
                    elements.append(el)
        # object properties
        for prop in list(g.subjects(RDF.type, OWL.ObjectProperty)):
            if isinstance(prop, URIRef):
                el = self._extract_property_info(g, prop, 'object_property', ontology_file)
                if el:
                    elements.append(el)
        # data properties
        for prop in list(g.subjects(RDF.type, OWL.DatatypeProperty)):
            if isinstance(prop, URIRef):
                el = self._extract_property_info(g, prop, 'data_property', ontology_file)
                if el:
                    elements.append(el)
        # annotation properties
        for prop in list(g.subjects(RDF.type, OWL.AnnotationProperty)):
            if isinstance(prop, URIRef):
                el = self._extract_property_info(g, prop, 'annotation_property', ontology_file)
                if el:
                    elements.append(el)

        # Named OWL restrictions (skip blank nodes)
        for restriction in list(g.subjects(RDF.type, OWL.Restriction)):
            el = self._extract_named_restriction(g, restriction, ontology_file)
            if el:
                elements.append(el)
        return elements

    @staticmethod
    def _extract_class_info(g: Graph, cls: URIRef, source_ontology: str) -> Optional[OntologyElement]:
        try:
            label = OntologyExtractor._get_literal_value(g, cls, RDFS.label)
            comment = OntologyExtractor._get_literal_value(g, cls, RDFS.comment)
            super_classes = [str(sc) for sc in g.objects(cls, RDFS.subClassOf) if isinstance(sc, URIRef)]
            sub_classes = [str(sc) for sc in g.subjects(RDFS.subClassOf, cls) if isinstance(sc, URIRef)]
            return OntologyElement(
                uri=str(cls),
                element_type='class',
                source_ontology=source_ontology,
                label=label,
                comment=comment,
                super_classes=super_classes,
                sub_classes=sub_classes,
                snippet=OntologyExtractor._collect_snippet(g, cls),
            )
        except Exception as e:
            print(f"[yellow] Error extracting class {cls}: {e}[/yellow]")
            return None

    @staticmethod
    def _extract_property_info(g: Graph, prop: URIRef, prop_type: str, source_ontology: str) -> Optional[OntologyElement]:
        try:
            label = OntologyExtractor._get_literal_value(g, prop, RDFS.label)
            comment = OntologyExtractor._get_literal_value(g, prop, RDFS.comment)
            domain = [str(d) for d in g.objects(prop, RDFS.domain) if isinstance(d, URIRef)]
            range_vals = [str(r) for r in g.objects(prop, RDFS.range) if isinstance(r, URIRef)]
            return OntologyElement(
                uri=str(prop),
                element_type=prop_type,
                source_ontology=source_ontology,
                label=label,
                comment=comment,
                domain=domain,
                range=range_vals,
                snippet=OntologyExtractor._collect_snippet(g, prop),
            )
        except Exception as e:
            print(f"[yellow]Error extracting property {prop}: {e}[/yellow]")
            return None

    @staticmethod
    def _get_literal_value(g: Graph, subject: URIRef, predicate: URIRef) -> Optional[str]:
        for obj in g.objects(subject, predicate):
            if isinstance(obj, Literal):
                return str(obj)
        return None

    @staticmethod
    def _local_name(uri: str) -> str:
        return uri.split('#')[-1].split('/')[-1]

    @staticmethod
    def _format_uri_list(values: List[str]) -> str:
        return ", ".join(OntologyExtractor._local_name(v) for v in values)

    @staticmethod
    def _extract_named_restriction(g: Graph, restriction: Any, source_ontology: str) -> Optional[OntologyElement]:
        if not isinstance(restriction, URIRef) or isinstance(restriction, BNode):
            return None
        on_property = next((str(op) for op in g.objects(restriction, OWL.onProperty) if isinstance(op, URIRef)), None)
        some_values = next((str(val) for val in g.objects(restriction, OWL.someValuesFrom) if isinstance(val, URIRef)), None)
        all_values = next((str(val) for val in g.objects(restriction, OWL.allValuesFrom) if isinstance(val, URIRef)), None)
        has_value = OntologyExtractor._get_literal_value(g, restriction, OWL.hasValue)
        info: List[str] = []
        if on_property:
            info.append(f"On Property: {OntologyExtractor._local_name(on_property)}")
        if some_values:
            info.append(f"Some Values From: {OntologyExtractor._local_name(some_values)}")
        if all_values:
            info.append(f"All Values From: {OntologyExtractor._local_name(all_values)}")
        if has_value:
            info.append(f"Has Value: {has_value}")
        for pred, label in [
            (OWL.cardinality, "Cardinality"),
            (OWL.minCardinality, "Min Cardinality"),
            (OWL.maxCardinality, "Max Cardinality"),
            (OWL.qualifiedCardinality, "Qualified Cardinality"),
            (OWL.minQualifiedCardinality, "Min Qualified Cardinality"),
            (OWL.maxQualifiedCardinality, "Max Qualified Cardinality"),
        ]:
            val = OntologyExtractor._get_literal_value(g, restriction, pred)
            if val:
                info.append(f"{label}: {val}")
        label_literal = OntologyExtractor._get_literal_value(g, restriction, RDFS.label)
        comment = OntologyExtractor._get_literal_value(g, restriction, RDFS.comment)
        return OntologyElement(
            uri=str(restriction),
            element_type='restriction',
            source_ontology=source_ontology,
            label=label_literal,
            comment=comment,
            additional_info=info,
            snippet=OntologyExtractor._collect_snippet(g, restriction),
        )


class FaissVectorStore:
    def __init__(self, dim: int, inner_product: bool = True) -> None:
        self.idx = faiss.IndexFlatIP(dim) if inner_product else faiss.IndexFlatL2(dim)
        self.elements: List[OntologyElement] = []
        self.seen_uris: Set[str] = set()

    def is_new(self, uri: str) -> bool:
        return uri not in self.seen_uris

    def add(self, vec: np.ndarray, element: OntologyElement) -> bool:
        if not element.uri or element.uri in self.seen_uris:
            return False
        self.idx.add(vec)
        self.elements.append(element)
        self.seen_uris.add(element.uri)
        return True

    def search(self, vec: np.ndarray, k: int = 10) -> List[Tuple[OntologyElement, float]]:
        if self.idx.ntotal == 0:
            return []
        D, I = self.idx.search(vec, k)
        return [(self.elements[i], float(D[0][j])) for j, i in enumerate(I[0]) if i != -1]


# ==========================
# Prompting & LLM
# ==========================

class PromptBuilder:
    def __init__(self, prefix_block: str) -> None:
        self.prefix_block = prefix_block

    @staticmethod
    def _format_relevant_elements(elements: List[OntologyElement]) -> Tuple[str, int]:
        elements_by_source: Dict[str, List[OntologyElement]] = {}
        for e in elements:
            src = Path(e.source_ontology).stem
            elements_by_source.setdefault(src, []).append(e)
        parts: List[str] = []
        for source, els in elements_by_source.items():
            parts.append(f"\n## From {source} ontology:")
            for e in els:
                parts.append(f"- {e.get_searchable_text()}")
                if e.snippet:
                    snippet = e.snippet.strip()
                    if snippet:
                        parts.append("```ttl")
                        parts.append(snippet)
                        parts.append("```")
                parts.append("_Reuse this element exactly as defined; do not redefine it._")
        return "\n".join(parts), len(elements_by_source)

    def build(self, cq: str, elements: List[OntologyElement], feedback: Optional[str] = None) -> str:
        relevant_text, num_ontologies = self._format_relevant_elements(elements)
        prompt = USER_TMPL.format(
            num_ontologies=num_ontologies,
            relevant_elements=relevant_text,
            cq=cq,
            prefix_block=self.prefix_block,
        )
        if feedback:
            prompt += f"\nPREVIOUS ATTEMPT ISSUES:\n- {feedback}\nPlease fix these issues before returning the ontology."
        return prompt


# ==========================
# Validation & Combination
# ==========================

class TurtleValidator:
    @staticmethod
    def validate_ttl(ttl_content: str) -> bool:
        try:
            g = Graph()
            g.parse(data=ttl_content, format="turtle")
            return True
        except Exception as e:
            print(f"[yellow]Turtle validation failed: {e}[/yellow]")
            return False


class OntologyCombiner:
    @staticmethod
    def remove_prefixes_from_fragment(fragment: str) -> str:
        lines = fragment.split('\n')
        cleaned: List[str] = []
        for line in lines:
            s = line.strip()
            if s.startswith('@prefix') or s.startswith('PREFIX'):
                continue
            cleaned.append(line)
        return '\n'.join(cleaned)

    @staticmethod
    def combine(fragments: List[Tuple[str, str]], ontology_files: List[str], prefix_block: str) -> str:
        header = (
            f"{prefix_block}\n"
            f"# Combined ontology generated from competency questions\n"
            f"# Reference ontologies: {', '.join(Path(f).name for f in ontology_files)}\n"
            f"# Generated on: {datetime.now().isoformat()}\n"
            f"# Total fragments: {len(fragments)}\n\n"
            "<http://www.example.org/ontology> a owl:Ontology .\n\n"
        )
        body: List[str] = []
        for i, (cq, fragment) in enumerate(fragments, 1):
            body.append(f"# ── Fragment {i}: {cq[:80]}{'...' if len(cq) > 80 else ''} ──\n")
            body.append(OntologyCombiner.remove_prefixes_from_fragment(fragment).strip())
            body.append("")
        return header + "\n".join(body)


# ==========================
# RAG Application
# ==========================

class OntologyRAG:
    def __init__(self, paths: Paths, normalize_vectors: bool = True, batch_size: int = 64) -> None:
        self.paths = paths
        self.tracker = TokenCostTracker()
        self.logger = RunLogger(paths, self.tracker)
        self.extractor = OntologyExtractor(self.logger)
        self.prefix_mgr = PrefixManager()
        self.embedder = get_embedder(EMBEDDER, EMBED_MODEL, self.tracker, normalize=normalize_vectors, batch_size=batch_size)
        self.store = FaissVectorStore(int(VECTOR_DIM), inner_product=True)  # cosine if normalized
        self.llm = get_llm_service(LLM_SERVICE, LLM_MODEL, LLM_TEMPERATURE, self.tracker)
        self.reviewer = FragmentReviewer()
        self.include_cache: bool = False
        self.reference_sources: Set[str] = set()
        self.session_sources: Set[str] = set()
        self.cache_sources: Set[str] = set()
        self.core_class_uris: Set[str] = set()
        self.core_property_uris: Set[str] = set()
        self.core_entities: Set[str] = set()

    @staticmethod
    def _resolve_path(path: str) -> str:
        try:
            return str(Path(path).resolve())
        except Exception:
            return path

    def _allowed_sources(self) -> Set[str]:
        allowed = set(self.reference_sources)
        allowed.update(self.session_sources)
        if self.include_cache:
            allowed.update(self.cache_sources)
        return allowed

    def _collect_named_entities(self, ontology_files: List[str]) -> Set[str]:
        entities: Set[str] = set()
        predicates = [
            (RDF.type, OWL.Class),
            (RDF.type, OWL.ObjectProperty),
            (RDF.type, OWL.DatatypeProperty),
            (RDF.type, OWL.AnnotationProperty),
            (RDF.type, RDF.Property),
        ]
        for onto_path in ontology_files:
            g = Graph()
            try:
                g.parse(onto_path)
            except Exception as exc:
                print(f"[yellow]Named-entity scan skipped for {onto_path}: {exc}[/yellow]")
                continue
            for predicate, obj in predicates:
                for subj in g.subjects(predicate, obj):
                    if isinstance(subj, URIRef):
                        entities.add(str(subj))
            for subj in g.subjects(RDFS.subClassOf, None):
                if isinstance(subj, URIRef):
                    entities.add(str(subj))
            for subj in g.subjects(RDFS.subPropertyOf, None):
                if isinstance(subj, URIRef):
                    entities.add(str(subj))
            for subj in g.subjects(RDFS.domain, None):
                if isinstance(subj, URIRef):
                    entities.add(str(subj))
            for subj in g.subjects(RDFS.range, None):
                if isinstance(subj, URIRef):
                    entities.add(str(subj))
            for subj in g.subjects(OWL.equivalentClass, None):
                if isinstance(subj, URIRef):
                    entities.add(str(subj))
            for subj in g.subjects(OWL.equivalentProperty, None):
                if isinstance(subj, URIRef):
                    entities.add(str(subj))
            for subj in g.subjects(RDFS.label, None):
                if isinstance(subj, URIRef):
                    entities.add(str(subj))
            for subj in g.subjects(RDFS.comment, None):
                if isinstance(subj, URIRef):
                    entities.add(str(subj))
        return entities

    @staticmethod
    def _sanitize_token(token: str, default: str = "item") -> str:
        cleaned = re.sub(r"[^A-Za-z0-9]+", "-", token.strip().lower())
        cleaned = re.sub(r"-+", "-", cleaned).strip("-")
        return cleaned or default

    def _core_tag(self, ontology_files: List[str]) -> str:
        if not ontology_files:
            return "core"
        stems = sorted({Path(f).stem for f in ontology_files})
        combined = "-".join(stems)
        return self._sanitize_token(combined, "core")

    def _next_sequence(self, onto_tag: str) -> int:
        pattern = re.compile(r"cq(\d+)_")
        sequences: List[int] = []
        targets = [
            self.paths.individual_dir,
            self.paths.retrieved_dir,
        ]
        for directory in targets:
            if not directory.exists():
                continue
            for path in directory.iterdir():
                if not path.is_file():
                    continue
                stem = path.stem
                if not (
                    stem.endswith(onto_tag)
                    or stem.endswith(f"{onto_tag}_retrieved")
                ):
                    continue
                match = pattern.search(stem)
                if match:
                    sequences.append(int(match.group(1)))
        return max(sequences, default=0) + 1

    def _build_cq_basename(self, cq_identifier: Optional[str], seq_num: int, onto_tag: str) -> str:
        parts: List[str] = []
        if cq_identifier:
            parts.append(self._sanitize_token(cq_identifier, "id"))
        parts.append(f"cq{seq_num:03d}")
        parts.append(onto_tag)
        return "_".join(parts)

    @staticmethod
    def _extract_cq_id(row: pd.Series) -> Optional[str]:
        candidate_cols = [
            "CQID", "cqid", "CQ_ID", "cq_id", "CQId", "CQ Id",
            "ID", "Id", "id"
        ]
        for col in candidate_cols:
            if col in row and pd.notna(row[col]):
                value = str(row[col]).strip()
                if value and value.lower() != "nan":
                    return value
        return None

    def _write_retrieved_elements(self, basename: str, elements: List[OntologyElement]) -> None:
        out_path = self.paths.retrieved_dir / f"{basename}_retrieved.csv"
        with out_path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow(["uri", "element_type", "source", "label", "domains", "ranges", "notes"])
            for el in elements:
                writer.writerow([
                    el.uri,
                    el.element_type,
                    Path(el.source_ontology).stem,
                    el.label or "",
                    ";".join(el.domain or []),
                    ";".join(el.range or []),
                    ";".join(el.additional_info or []),
                ])
        print(f"[blue]Retrieved elements saved to: {out_path}[/blue]")

    async def _index_elements(self, elements: List[OntologyElement]) -> None:
        new_elements = [e for e in elements if self.store.is_new(e.uri)]
        if not new_elements:
            return
        texts = [e.get_searchable_text() for e in new_elements]
        vecs = await self.embedder.embed_many(texts)
        for vec, element in zip(vecs, new_elements):
            self.store.add(vec, element)

    async def _index_additional_files(self, files: List[str], *, origin: Literal["cache", "session"]) -> None:
        if not files:
            return
        elements = self.extractor.extract_from_files(files, log=False)
        await self._index_elements(elements)
        resolved = {self._resolve_path(f) for f in files}
        if origin == "cache":
            self.cache_sources.update(resolved)
        else:
            self.session_sources.update(resolved)

    async def _index_existing_fragments(self) -> None:
        if not self.include_cache:
            return
        if not self.paths.individual_dir.exists():
            return
        cache_files = sorted(self.paths.individual_dir.glob("*.ttl"))
        if not cache_files:
            return
        print(f"[yellow]Loading {len(cache_files)} cached CQ fragments into the vector store...[/yellow]")
        await self._index_additional_files([str(f) for f in cache_files], origin="cache")

    # @staticmethod
    # def ensure_api_key() -> None:
    #     if not os.getenv("OPENAI_API_KEY"):
    #         print("[bold red]OPENAI_API_KEY is not set[/bold red]")
    #         sys.exit(1)

    async def _build_store(self, ontology_files: List[str]) -> str:
        print("[yellow]Building vector store from reference ontologies...[/yellow]")
        self.reference_sources = {self._resolve_path(f) for f in ontology_files}
        self.session_sources.clear()
        if not self.include_cache:
            self.cache_sources.clear()
        elements = self.extractor.extract_from_files(ontology_files)
        self.core_class_uris = {el.uri for el in elements if getattr(el, "element_type", "") == "class" and el.uri}
        self.core_property_uris = {
            el.uri
            for el in elements
            if getattr(el, "element_type", "") in {"object_property", "data_property", "annotation_property"}
            and el.uri
        }
        self.core_entities = self._collect_named_entities(ontology_files)
        await self._index_elements(elements)
        await self._index_existing_fragments()
        prefix_block = self.prefix_mgr.build_prefix_block(ontology_files)
        print(f"[green]Indexed {len(self.store.elements)} ontology elements from {len(ontology_files)} ontologies[/green]")
        return prefix_block

    async def _retrieve_relevant(self, cq: str, k: int = 20) -> List[OntologyElement]:
        qvec = await self.embedder.embed_one(cq)
        results = self.store.search(qvec, k)
        allowed_sources = self._allowed_sources()
        filtered: List[OntologyElement] = []
        for el, _ in results:
            src = self._resolve_path(el.source_ontology)
            if src in allowed_sources:
                filtered.append(el)
        return filtered

    async def _generate_fragment(self, cq: str, relevant: List[OntologyElement], prefix_block: str, feedback: Optional[str] = None) -> str:
        prompt = PromptBuilder(prefix_block).build(cq, relevant, feedback=feedback)
        response = await self.llm.complete(prompt, system_prompt=SYS_PROMPT)
        m = re.search(r"```(?:ttl|turtle)?\n(.*?)```", response, re.DOTALL)
        fragment = m.group(1).strip() if m else response.strip()
        return fragment

    async def _generate_with_retries(
        self,
        cq: str,
        relevant: List[OntologyElement],
        prefix_block: str,
    ) -> FragmentReviewResult:
        fragment = await self._generate_fragment(cq, relevant, prefix_block)
        return self.reviewer.review(fragment, cq, relevant)

    @staticmethod
    def _report_review_failure(attempt: int, review: FragmentReviewResult) -> None:
        issue_text = ", ".join(review.issues) if review.issues else "unspecified issues"
        message = f"Attempt {attempt} rejected due to: {issue_text}."
        if review.feedback_hint:
            message += f" Hint: {review.feedback_hint}"
        offenders = review.details.get("redefinition") if isinstance(review.details, dict) else None
        if offenders:
            message += f" Offending IRIs: {', '.join(offenders.keys())}."
        print(f"[yellow]{message} Retrying with adjusted instructions...[/yellow]")

    def _cleanup_fragment(self, fragment: str) -> str:
        """Remove core-defined entity declarations and orphaned blank nodes from the generated fragment."""
        if not fragment.strip() or not self.core_entities:
            return fragment
        try:
            graph = Graph()
            graph.parse(data=fragment, format="turtle")
        except Exception as exc:
            print(f"[yellow]Cleanup skipped; fragment failed to parse: {exc}[/yellow]")
            return fragment

        protected = {URIRef(uri) for uri in self.core_entities}
        overlapping: Set[URIRef] = set()
        for uri in protected:
            if any(True for _ in graph.triples((uri, None, None))):
                overlapping.add(uri)
        if not overlapping:
            return fragment

        triple_removals = 0
        for uri in overlapping:
            for triple in list(graph.triples((uri, None, None))):
                graph.remove(triple)
                triple_removals += 1

        def prune_blank_nodes(g: Graph) -> int:
            total_removed = 0
            while True:
                blank_nodes: Set[BNode] = set()
                for s, p, o in g:
                    if isinstance(s, BNode):
                        blank_nodes.add(s)
                    if isinstance(o, BNode):
                        blank_nodes.add(o)
                if not blank_nodes:
                    break
                to_remove: Set[BNode] = set()
                for bnode in blank_nodes:
                    if not any(not isinstance(s, BNode) for s, _, _ in g.triples((None, None, bnode))):
                        to_remove.add(bnode)
                if not to_remove:
                    break
                for s, p, o in list(g):
                    if (isinstance(s, BNode) and s in to_remove) or (isinstance(o, BNode) and o in to_remove):
                        g.remove((s, p, o))
                        total_removed += 1
            return total_removed

        blank_removed = prune_blank_nodes(graph)
        cleaned = graph.serialize(format="turtle")
        print(
            f"[blue]Cleanup removed {len(overlapping)} core entities, {triple_removals} triples, {blank_removed} blank-node triples.[/blue]"
        )
        return cleaned if isinstance(cleaned, str) else cleaned.decode("utf-8")

    async def run_single_cq(self, cq: str, ontology_files: List[str], top_k: int = 20) -> None:
        prefix_block = await self._build_store(ontology_files)
        onto_tag = self._core_tag(ontology_files)
        seq_num = self._next_sequence(onto_tag)
        basename = self._build_cq_basename(None, seq_num, onto_tag)
        before = self.tracker.snapshot()
        relevant = await self._retrieve_relevant(cq, k=top_k)
        self._write_retrieved_elements(basename, relevant)
        try:
            review = await self._generate_with_retries(cq, relevant, prefix_block)
            fragment = review.fragment
            if not review.accepted:
                print("[yellow]Proceeding with the last fragment even though validation issues remain.[/yellow]")
            fragment = self._cleanup_fragment(fragment)
        except ValueError as err:
            print(f"[red]{err}[/red]")
            after = self.tracker.snapshot()
            self.logger.log_cost_row(-1, cq, self.tracker.diff(before, after))
            return
        out_path = self.paths.individual_dir / f"{basename}.ttl"
        out_path.write_text(fragment, encoding="utf-8")
        print(f"[green]Ontology fragment written to: {out_path}[/green]")
        is_valid = TurtleValidator.validate_ttl(fragment)
        if is_valid:
            await self._index_additional_files([str(out_path)], origin="session")
            print("[green]Fragment passed Turtle validation.[/green]")
        else:
            print("[bold yellow]Generated fragment failed Turtle validation (still saved).[/bold yellow]")
        print("\n[bold green]Ontology fragment[/bold green]\n")
        print(fragment)
        after = self.tracker.snapshot()
        self.logger.log_cost_row(-1, cq, self.tracker.diff(before, after))
        print(f"[bold green]💲 Estimated total API cost (USD): {self.tracker.totals_cost_usd():.6f}[/bold green]")

    async def run_csv(self, csv_file: str, ontology_files: List[str], limit: Optional[int] = None, top_k: int = 20) -> None:
        print(f"[bold blue]Processing CQ dataset: {csv_file}[/bold blue]")
        print(f"[bold blue]Using {len(ontology_files)} reference ontologies:[/bold blue]")
        for i, onto_file in enumerate(ontology_files, 1):
            print(f"[cyan]  {i}. {Path(onto_file).name}[/cyan]")

        if not Path(csv_file).exists():
            print(f"[red]CSV file not found: {csv_file}[/red]")
            return
        missing = [f for f in ontology_files if not Path(f).exists()]
        if missing:
            print(f"[red]Ontology files not found: {missing}[/red]")
            return

        try:
            df = pd.read_csv(csv_file)
            print(f"[green]Loaded {len(df)} rows from CSV[/green]")
        except Exception as e:
            print(f"[red]Error reading CSV: {e}[/red]")
            return

        if 'CQ' not in df.columns:
            print(f"[red]'CQ' column not found in CSV. Available columns: {list(df.columns)}[/red]")
            return

        if limit:
            df = df.head(limit)
            print(f"[blue]Processing limited to first {len(df)} rows[/blue]")

        prefix_block = await self._build_store(ontology_files)
        onto_tag = self._core_tag(ontology_files)

        fragments: List[Tuple[str, str]] = []
        successful = 0
        failed = 0
        start_time = time.time()

        from tqdm import tqdm  # local import to avoid mandatory dep during static analysis
        pbar = tqdm(total=len(df), desc="Processing CQs")

        for seq_num, (idx, row) in enumerate(df.iterrows(), start=1):
            cq = str(row['CQ']).strip()
            before = self.tracker.snapshot()

            if not cq or cq.lower() in ['nan', 'none', '']:
                print(f"[yellow]Row {idx}: Empty CQ, skipping[/yellow]")
                self.logger.log_processing_result(idx, cq, False, 0, {}, 0, "Empty CQ")
                failed += 1
                pbar.update(1)
                continue

            pbar.set_description(f"Processing CQ {idx}")
            cq_start = time.time()
            cq_identifier = self._extract_cq_id(row)
            basename = self._build_cq_basename(cq_identifier, seq_num, onto_tag)

            try:
                relevant = await self._retrieve_relevant(cq, k=top_k)
                self._write_retrieved_elements(basename, relevant)
                # counts by source (for logging)
                src_counts: Dict[str, int] = {}
                for el in relevant:
                    src_counts[Path(el.source_ontology).stem] = src_counts.get(Path(el.source_ontology).stem, 0) + 1

                try:
                    review = await self._generate_with_retries(cq, relevant, prefix_block)
                    fragment = review.fragment
                    if not review.accepted:
                        print("[yellow]Proceeding with the last fragment even though validation issues remain.[/yellow]")
                    fragment = self._cleanup_fragment(fragment)
                except ValueError as err:
                    print(f"[red]{err}[/red]")
                    self.logger.log_processing_result(idx, cq, False, len(relevant), src_counts, time.time() - cq_start, str(err))
                    failed += 1
                    continue
                out_file = self.paths.individual_dir / f"{basename}.ttl"
                out_file.write_text(fragment, encoding="utf-8")
                is_valid = TurtleValidator.validate_ttl(fragment)
                if is_valid:
                    fragments.append((cq, fragment))
                    await self._index_additional_files([str(out_file)], origin="session")
                    self.logger.log_processing_result(idx, cq, True, len(relevant), src_counts, time.time() - cq_start)
                    successful += 1
                    print(f"[green]CQ {idx} processed successfully[/green]")
                else:
                    print(f"[yellow]CQ {idx}: Fragment saved but failed Turtle validation[/yellow]")
                    self.logger.log_processing_result(idx, cq, False, len(relevant), src_counts, time.time() - cq_start, "Invalid Turtle")
                    failed += 1
            except Exception as e:
                print(f"[red]CQ {idx} failed: {e}[/red]")
                self.logger.log_processing_result(idx, cq, False, 0, {}, time.time() - cq_start, str(e))
                failed += 1
            finally:
                after = self.tracker.snapshot()
                self.logger.log_cost_row(idx, cq, self.tracker.diff(before, after))
                pbar.update(1)
                await asyncio.sleep(0.1)

        pbar.close()

        if fragments:
            print(f"[yellow]Combining {len(fragments)} ontology fragments...[/yellow]")
            combined = OntologyCombiner.combine(fragments, ontology_files, prefix_block)
            combined_basename = f"{Path(csv_file).stem}_{onto_tag}"
            combined_path = self.paths.cache_dir / f"{combined_basename}_combined.ttl"
            combined_path.write_text(combined, encoding="utf-8")
            print(f"[green]Combined ontology saved to: {combined_path}[/green]")
            if TurtleValidator.validate_ttl(combined):
                print("[green]Combined ontology is syntactically valid[/green]")
            else:
                print("[yellow]Combined ontology has syntax issues[/yellow]")
        else:
            combined_path = self.paths.combined_ttl

        total_time = time.time() - start_time
        self.logger.log_final_results(len(df), successful, failed, total_time, str(combined_path))

        print(f"\n[bold green]Processing Complete![/bold green]")
        print(f"Total CQs: {len(df)}")
        print(f"Successful: {successful}")
        print(f"Failed: {failed}")
        print(f"Success rate: {(successful / len(df) * 100):.1f}%")
        print(f"[bold green]💲 Estimated total API cost (USD): {self.tracker.totals_cost_usd():.6f}[/bold green]")


# ==========================
# CLI
# ==========================


def expand_onto_args(files: Optional[List[str]], dirs: Optional[List[str]]) -> List[str]:
    candidates: List[str] = []
    if files:
        candidates.extend(files)
    if dirs:
        for d in dirs:
            p = Path(d)
            if not p.exists() or not p.is_dir():
                print(f"[yellow]Not a directory (skipped): {d}[/yellow]")
                continue
            candidates.extend(str(f) for f in p.rglob("*.ttl"))
            candidates.extend(str(f) for f in p.rglob("*.owl"))
    seen = set()
    result: List[str] = []
    for f in candidates:
        if f not in seen:
            result.append(f)
            seen.add(f)
    if not result:
        print("[red]No ontology files found from --onto/--onto-dir[/red]")
        sys.exit(1)
    return result


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Core ontology RAG for CQ → TTL (tidied)")
    sub = p.add_subparsers(dest="cmd")

    p_single = sub.add_parser("cq", help="Generate ontology for a single CQ")
    p_single.add_argument("--cq", required=True, help="Competency Question text")
    p_single.add_argument("--onto", nargs="+", help="One or more reference ontology files (.ttl/.owl)")
    p_single.add_argument("--onto-dir", nargs="+", help="One or more directories to scan for .ttl/.owl")
    p_single.add_argument("--top-k", type=int, default=20, help="Top-K elements to retrieve")
    p_single.add_argument("--include-cache", action="store_true", help="Also reuse previously generated CQ fragments from cache")

    p_batch = sub.add_parser("csv", help="Process a CSV with a CQ column")
    p_batch.add_argument("--file", required=True, help="CSV file with column 'CQ'")
    p_batch.add_argument("--onto", nargs="+", help="One or more reference ontology files (.ttl/.owl)")
    p_batch.add_argument("--onto-dir", nargs="+", help="One or more directories to scan for .ttl/.owl")
    p_batch.add_argument("--limit", type=int, default=None, help="Process only first N rows")
    p_batch.add_argument("--top-k", type=int, default=20, help="Top-K elements to retrieve")
    p_batch.add_argument("--include-cache", action="store_true", help="Also reuse previously generated CQ fragments from cache")

    return p.parse_args()


async def _main_async(args: argparse.Namespace) -> None:
    paths = Paths()
    app = OntologyRAG(paths)

    if not args.cmd:
        print("[bold red]Usage:[/bold red] python tidied_rag_ontology.py cq --cq \"...\" --onto core.ttl more.ttl")
        print("         python tidied_rag_ontology.py csv --file dataset.csv --onto core.ttl more.ttl [--limit 50]")
        sys.exit(1)

    if args.cmd == "cq":
        app.include_cache = bool(getattr(args, "include_cache", False))
        ontofiles = expand_onto_args(args.onto, args.onto_dir)
        await app.run_single_cq(args.cq, ontofiles, top_k=args.top_k)
    elif args.cmd == "csv":
        app.include_cache = bool(getattr(args, "include_cache", False))
        ontofiles = expand_onto_args(args.onto, args.onto_dir)
        await app.run_csv(args.file, ontofiles, limit=args.limit, top_k=args.top_k)


def main() -> None:
    args = parse_args()
    asyncio.run(_main_async(args))


if __name__ == "__main__":
    main()


#python filename.py csv --file .\cqs.csv --onto-dir .\coreontologies --limit 3
