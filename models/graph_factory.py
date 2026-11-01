"""
Loading and saving road maps (Phase 1, requirement F1).

GraphFactory turns a JSON or CSV file into a RoadGraph, a VehicleFleet and a
DynamicJobQueue, and writes them back to a file. Every value in a file is
checked before it is used, so a bad file raises InvalidGraphFileError with a
message saying where the problem is, and never crashes the program.

JSON layout (version 1):

    {
      "version": 1,
      "distance_unit": "km",
      "nodes":    [{"id": "Depot", "x": 0, "y": 0, "name": "Main depot", "type": "depot"}],
      "edges":    [{"from": "Depot", "to": "A", "weight": 4.5, "directed": false}],
      "vehicles": [{"id": "V1", "capacity": 100, "speed": 50,
                    "fuel_rate": 0.1, "home_depot": "Depot"}],
      "jobs":     [{"id": "J1", "pickup": "A", "dropoff": "B", "weight": 10, "deadline": 5}]
    }

"nodes" and "edges" are required. "vehicles", "jobs", "version",
"distance_unit", and each node's "name", "type" and each edge's "directed"
are optional.

CSV layout: one file, one record per row, with the kind of record in the
first column. Columns that do not apply to a record are left empty.

    record,id,x,y,name,type,from,to,weight,directed,capacity,speed,fuel_rate,home_depot,pickup,dropoff,deadline,value
    unit,,,,,,,,,,,,,,,,,km
    node,Depot,0,0,Main depot,depot,,,,,,,,,,,,
    node,A,3,4,,,,,,,,,,,,,,
    edge,,,,,,Depot,A,5,false,,,,,,,,
    vehicle,V1,,,,,,,,,100,50,0.1,Depot,,,,
    job,J1,,,,,,,10,,,,,,A,B,5,

Both formats are checked by the same rules.
"""

import csv
import io
import json
import os
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional, Union

from .graph import Node, NodeType, RoadGraph
from .job import DynamicJobQueue, Job
from .vehicle import Vehicle, VehicleFleet

FORMAT_VERSION = 1
MAX_FILE_BYTES = 10 * 1024 * 1024  # refuse files over 10 MB instead of loading them

PathLike = Union[str, "os.PathLike[str]"]

CSV_COLUMNS = (
    "record", "id", "x", "y", "name", "type",
    "from", "to", "weight", "directed",
    "capacity", "speed", "fuel_rate", "home_depot",
    "pickup", "dropoff", "deadline", "value",
)


class InvalidGraphFileError(ValueError):
    """Raised when a map file is missing, unreadable, or contains bad data."""


@dataclass
class ProjectData:
    """Everything a map file describes."""

    graph: RoadGraph
    fleet: VehicleFleet
    jobs: DynamicJobQueue
    warnings: List[str] = field(default_factory=list)  # problems that do not stop loading


@contextmanager
def _context(where: str) -> Iterator[None]:
    """Turn errors raised inside the block into InvalidGraphFileError mentioning `where`."""
    try:
        yield
    except InvalidGraphFileError:
        raise
    except (ValueError, KeyError, TypeError) as error:
        message = error.args[0] if isinstance(error, KeyError) and error.args else str(error)
        raise InvalidGraphFileError(f"{where}: {message}") from error


def _reject_constant(name: str) -> None:
    raise ValueError(f"'{name}' is not allowed; numbers must be finite.")


# --- converters for CSV cells (every cell arrives as text) ---------------

def _as_text(text: str) -> str:
    return text


def _as_number(text: str) -> float:
    try:
        return float(text)
    except ValueError:
        raise ValueError(f"'{text}' is not a number") from None


def _as_flag(text: str) -> bool:
    lowered = text.lower()
    if lowered in ("true", "yes", "1"):
        return True
    if lowered in ("false", "no", "0"):
        return False
    raise ValueError(f"'{text}' is not true or false")


# For each kind of CSV record: the JSON section it belongs to, and how to
# convert each column that it uses.
_CSV_RECORDS: Dict[str, Dict[str, Any]] = {
    "node": {
        "section": "nodes",
        "columns": {"id": _as_text, "x": _as_number, "y": _as_number,
                    "name": _as_text, "type": _as_text},
    },
    "edge": {
        "section": "edges",
        "columns": {"from": _as_text, "to": _as_text, "weight": _as_number,
                    "directed": _as_flag},
    },
    "vehicle": {
        "section": "vehicles",
        "columns": {"id": _as_text, "capacity": _as_number, "speed": _as_number,
                    "fuel_rate": _as_number, "home_depot": _as_text},
    },
    "job": {
        "section": "jobs",
        "columns": {"id": _as_text, "pickup": _as_text, "dropoff": _as_text,
                    "weight": _as_number, "deadline": _as_number},
    },
}


class GraphFactory:
    """Reads and writes map files. All methods are class methods; no instance is needed."""

    # ------------------------------------------------------------------
    # Choosing the format from the file name
    # ------------------------------------------------------------------

    @classmethod
    def load(cls, path: PathLike) -> ProjectData:
        """Load a .json or .csv file, chosen by the file extension."""
        suffix = Path(path).suffix.lower()
        if suffix == ".json":
            return cls.load_json(path)
        if suffix == ".csv":
            return cls.load_csv(path)
        raise InvalidGraphFileError(
            f"Unsupported file type '{suffix}'; use a .json or .csv file."
        )

    @classmethod
    def save(
        cls,
        path: PathLike,
        graph: RoadGraph,
        fleet: Optional[VehicleFleet] = None,
        jobs: Optional[DynamicJobQueue] = None,
    ) -> Path:
        """Save to a .json or .csv file, chosen by the file extension."""
        suffix = Path(path).suffix.lower()
        if suffix == ".json":
            return cls.save_json(path, graph, fleet, jobs)
        if suffix == ".csv":
            return cls.save_csv(path, graph, fleet, jobs)
        raise InvalidGraphFileError(
            f"Unsupported file type '{suffix}'; use a .json or .csv file."
        )

    # ------------------------------------------------------------------
    # JSON files
    # ------------------------------------------------------------------

    @classmethod
    def load_json(cls, path: PathLike) -> ProjectData:
        """Read and validate a JSON map file."""
        text = cls._read_text(path)
        try:
            data = json.loads(text, parse_constant=_reject_constant)
        except json.JSONDecodeError as error:
            raise InvalidGraphFileError(
                f"Invalid JSON at line {error.lineno}, column {error.colno}: {error.msg}."
            ) from error
        except ValueError as error:  # raised by _reject_constant
            raise InvalidGraphFileError(f"Invalid JSON: {error}") from error

        return cls.from_dict(data)

    @classmethod
    def save_json(
        cls,
        path: PathLike,
        graph: RoadGraph,
        fleet: Optional[VehicleFleet] = None,
        jobs: Optional[DynamicJobQueue] = None,
    ) -> Path:
        """
        Write the map (and optionally the fleet and jobs) to a JSON file.
        An existing file at `path` is replaced; see _write_atomically.
        """
        document = cls.to_dict(graph, fleet, jobs)
        return cls._write_atomically(path, json.dumps(document, indent=2), "utf-8")

    # ------------------------------------------------------------------
    # CSV files
    # ------------------------------------------------------------------

    @classmethod
    def load_csv(cls, path: PathLike) -> ProjectData:
        """Read and validate a CSV map file (layout described at the top of this module)."""
        text = cls._read_text(path)
        return cls.from_dict(cls._csv_to_dict(text))

    @classmethod
    def save_csv(
        cls,
        path: PathLike,
        graph: RoadGraph,
        fleet: Optional[VehicleFleet] = None,
        jobs: Optional[DynamicJobQueue] = None,
    ) -> Path:
        """
        Write the map (and optionally the fleet and jobs) to a CSV file.
        The file starts with a byte-order mark so Excel shows accented
        characters correctly. An existing file at `path` is replaced.
        """
        document = cls.to_dict(graph, fleet, jobs)
        buffer = io.StringIO()
        writer = csv.DictWriter(buffer, fieldnames=CSV_COLUMNS, lineterminator="\r\n")
        writer.writeheader()

        writer.writerow({"record": "unit", "value": document["distance_unit"]})
        for record, spec in _CSV_RECORDS.items():
            for entry in document[spec["section"]]:
                row = {"record": record}
                for column in spec["columns"]:
                    # In the dictionary form an edge is stored under "from"/"to"
                    # and a node's type under "type", the same names as the columns.
                    value = entry[column]
                    row[column] = str(value).lower() if isinstance(value, bool) else value
                writer.writerow(row)
        return cls._write_atomically(path, buffer.getvalue(), "utf-8-sig")

    @classmethod
    def _csv_to_dict(cls, text: str) -> Dict[str, Any]:
        """Turn CSV text into the same dictionary layout a JSON file uses."""
        data: Dict[str, Any] = {"nodes": [], "edges": [], "vehicles": [], "jobs": []}
        try:
            reader = csv.DictReader(io.StringIO(text))
            if reader.fieldnames is None:
                raise InvalidGraphFileError("The CSV file is empty.")
            reader.fieldnames = [(name or "").strip().lower() for name in reader.fieldnames]
            if "record" not in reader.fieldnames:
                raise InvalidGraphFileError("The CSV file needs a 'record' column in its header row.")

            unit_seen = False
            for row in reader:
                where = f"line {reader.line_num}"
                if None in row:
                    raise InvalidGraphFileError(f"{where}: the row has more cells than the header.")
                cells = {key: (value or "").strip() for key, value in row.items()}
                if not any(cells.values()):
                    continue  # blank line

                record = cells["record"].lower()
                if record == "unit":
                    if unit_seen:
                        raise InvalidGraphFileError(f"{where}: the distance unit is given twice.")
                    unit_seen = True
                    data["distance_unit"] = cells.get("value", "")
                    continue
                if record not in _CSV_RECORDS:
                    allowed = ", ".join(["unit"] + list(_CSV_RECORDS))
                    raise InvalidGraphFileError(
                        f"{where}: unknown record type '{cells['record']}'; use one of: {allowed}."
                    )

                spec = _CSV_RECORDS[record]
                entry: Dict[str, Any] = {}
                for column, convert in spec["columns"].items():
                    text_value = cells.get(column, "")
                    if text_value == "":
                        continue  # left out, so defaults apply or "missing field" is reported
                    try:
                        entry[column] = convert(text_value)
                    except ValueError as error:
                        raise InvalidGraphFileError(
                            f"{where}: column '{column}': {error}."
                        ) from None
                data[spec["section"]].append(entry)
        except csv.Error as error:
            raise InvalidGraphFileError(f"Invalid CSV: {error}") from error
        return data

    # ------------------------------------------------------------------
    # Reading and writing files safely
    # ------------------------------------------------------------------

    @staticmethod
    def _read_text(path: PathLike) -> str:
        """Read a text file, refusing missing, huge or non-text files."""
        file_path = Path(path)
        try:
            if not file_path.is_file():
                raise InvalidGraphFileError(f"File not found: {file_path}")
            if file_path.stat().st_size > MAX_FILE_BYTES:
                raise InvalidGraphFileError(
                    f"File is too large (over {MAX_FILE_BYTES // (1024 * 1024)} MB): {file_path}"
                )
            return file_path.read_text(encoding="utf-8-sig")
        except UnicodeDecodeError as error:
            raise InvalidGraphFileError(f"File is not valid UTF-8 text: {file_path}") from error
        except OSError as error:
            raise InvalidGraphFileError(f"Could not read file {file_path}: {error}") from error

    @staticmethod
    def _write_atomically(path: PathLike, text: str, encoding: str) -> Path:
        """
        Write text to a temporary file and then move it into place, so a
        failure part-way through never destroys an existing file.
        """
        target = Path(path)
        temporary = target.with_name(target.name + ".tmp")
        try:
            temporary.write_bytes(text.encode(encoding))
            os.replace(temporary, target)
        except OSError as error:
            try:
                temporary.unlink()
            except OSError:
                pass
            raise OSError(f"Could not save map file '{target}': {error}") from error
        return target

    # ------------------------------------------------------------------
    # Dictionary form (shared by every file format)
    # ------------------------------------------------------------------

    @classmethod
    def to_dict(
        cls,
        graph: RoadGraph,
        fleet: Optional[VehicleFleet] = None,
        jobs: Optional[DynamicJobQueue] = None,
    ) -> Dict[str, Any]:
        """Describe the map, fleet and jobs as plain dictionaries and lists."""
        return {
            "version": FORMAT_VERSION,
            "distance_unit": graph.distance_unit,
            "nodes": [
                {
                    "id": node.id,
                    "x": node.x,
                    "y": node.y,
                    "name": node.name,
                    "type": node.node_type.value,
                }
                for node in graph.nodes.values()
            ],
            "edges": [
                {
                    "from": edge.source,
                    "to": edge.target,
                    "weight": edge.weight,
                    "directed": edge.directed,
                }
                for edge in graph.edges
            ],
            "vehicles": [
                {
                    "id": vehicle.id,
                    "capacity": vehicle.capacity,
                    "speed": vehicle.speed,
                    "fuel_rate": vehicle.fuel_rate,
                    "home_depot": vehicle.home_depot,
                }
                for vehicle in (fleet.vehicles if fleet is not None else ())
            ],
            "jobs": [
                {
                    "id": job.id,
                    "pickup": job.pickup,
                    "dropoff": job.dropoff,
                    "weight": job.weight,
                    "deadline": job.deadline,
                }
                for job in (jobs.all_jobs() if jobs is not None else ())
            ],
        }

    @classmethod
    def from_dict(cls, data: Any) -> ProjectData:
        """Validate a dictionary in the file layout and build the model objects."""
        if not isinstance(data, dict):
            raise InvalidGraphFileError("The file must contain a JSON object at the top level.")
        if "version" in data and data["version"] != FORMAT_VERSION:
            raise InvalidGraphFileError(
                f"Unsupported file version {data['version']!r}; expected {FORMAT_VERSION}."
            )

        with _context("distance_unit"):
            graph = RoadGraph(distance_unit=data.get("distance_unit", "km"))

        cls._load_nodes(graph, cls._list(data, "nodes", required=True))
        cls._load_edges(graph, cls._list(data, "edges", required=True))

        fleet = cls._load_vehicles(cls._list(data, "vehicles", required=False))
        problems = fleet.check_depots(graph)
        if problems:
            raise InvalidGraphFileError("vehicles: " + " ".join(problems))

        jobs = cls._load_jobs(cls._list(data, "jobs", required=False))
        problems = jobs.check_against(graph)
        if problems:
            raise InvalidGraphFileError("jobs: " + " ".join(problems))

        warnings = jobs.check_against_fleet(fleet) if len(fleet) and len(jobs) else []
        return ProjectData(graph=graph, fleet=fleet, jobs=jobs, warnings=warnings)

    # ------------------------------------------------------------------
    # Building each section
    # ------------------------------------------------------------------

    @classmethod
    def _load_nodes(cls, graph: RoadGraph, entries: List[Any]) -> None:
        for index, entry in enumerate(entries):
            where = f"nodes[{index}]"
            cls._require_dict(entry, where)
            node_id = cls._get(entry, "id", where)
            x = cls._get(entry, "x", where)
            y = cls._get(entry, "y", where)
            name = entry.get("name", "")
            if not isinstance(name, str):
                raise InvalidGraphFileError(f"{where}: 'name' must be text.")
            type_text = entry.get("type", NodeType.JUNCTION.value)
            allowed = ", ".join(t.value for t in NodeType)
            with _context(where):
                try:
                    node_type = NodeType(type_text)
                except ValueError:
                    raise ValueError(f"'type' must be one of: {allowed}.") from None
                graph.add_node(Node(node_id, x, y, name, node_type))

    @classmethod
    def _load_edges(cls, graph: RoadGraph, entries: List[Any]) -> None:
        for index, entry in enumerate(entries):
            where = f"edges[{index}]"
            cls._require_dict(entry, where)
            source = cls._get(entry, "from", where)
            target = cls._get(entry, "to", where)
            weight = cls._get(entry, "weight", where)
            directed = entry.get("directed", False)
            if not isinstance(directed, bool):
                raise InvalidGraphFileError(f"{where}: 'directed' must be true or false.")
            with _context(where):
                already = graph.has_edge(source, target) or (
                    not directed and graph.has_edge(target, source)
                )
                if already:
                    raise ValueError(f"a road between '{source}' and '{target}' already exists.")
                graph.add_edge(source, target, weight, directed)

    @classmethod
    def _load_vehicles(cls, entries: List[Any]) -> VehicleFleet:
        fleet = VehicleFleet()
        for index, entry in enumerate(entries):
            where = f"vehicles[{index}]"
            cls._require_dict(entry, where)
            values = [
                cls._get(entry, key, where)
                for key in ("id", "capacity", "speed", "fuel_rate", "home_depot")
            ]
            with _context(where):
                fleet.add(Vehicle(*values))
        return fleet

    @classmethod
    def _load_jobs(cls, entries: List[Any]) -> DynamicJobQueue:
        jobs = DynamicJobQueue()
        for index, entry in enumerate(entries):
            where = f"jobs[{index}]"
            cls._require_dict(entry, where)
            values = [
                cls._get(entry, key, where)
                for key in ("id", "pickup", "dropoff", "weight", "deadline")
            ]
            with _context(where):
                jobs.add(Job(*values))
        return jobs

    # ------------------------------------------------------------------
    # Small checks
    # ------------------------------------------------------------------

    @staticmethod
    def _list(data: Dict[str, Any], key: str, required: bool) -> List[Any]:
        if key not in data:
            if required:
                raise InvalidGraphFileError(f"Missing required section '{key}'.")
            return []
        if not isinstance(data[key], list):
            raise InvalidGraphFileError(f"Section '{key}' must be a list.")
        return data[key]

    @staticmethod
    def _require_dict(entry: Any, where: str) -> None:
        if not isinstance(entry, dict):
            raise InvalidGraphFileError(f"{where}: each entry must be an object.")

    @staticmethod
    def _get(entry: Dict[str, Any], key: str, where: str) -> Any:
        if key not in entry:
            raise InvalidGraphFileError(f"{where}: missing required field '{key}'.")
        return entry[key]