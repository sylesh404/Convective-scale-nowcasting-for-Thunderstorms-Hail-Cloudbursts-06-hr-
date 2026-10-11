"""
GIS Impact Engine Geometric Verification Suite (SIH26084 Part 7).
Adheres strictly to Section 15 of Part 7 Specification:
- Validates GIS polygon intersection calculations independently from weather prediction:
  1. Storm polygon + School point layer -> Affected schools
  2. Storm polygon + Hospital point layer -> Affected hospitals
  3. Storm polygon + Highway line segment layer -> Affected road segments
  4. Storm polygon + Population density grid -> Exposed population estimate
- Benchmarked against deterministic ground-truth geometric test vectors.
- Clearly states: "This is software validation, not disaster prediction validation."
"""

import math
from typing import Dict, Any, List, Tuple


class Point:
    def __init__(self, lat: float, lon: float, name: str = "", metadata: Dict[str, Any] = None):
        self.lat = lat
        self.lon = lon
        self.name = name
        self.metadata = metadata or {}


class Polygon:
    def __init__(self, vertices: List[Tuple[float, float]]):
        """vertices as [(lat, lon), ...]"""
        self.vertices = vertices

    def contains_point(self, point: Point) -> bool:
        """Standard ray casting point-in-polygon algorithm."""
        n = len(self.vertices)
        inside = False
        p1_lat, p1_lon = self.vertices[0]
        for i in range(1, n + 1):
            p2_lat, p2_lon = self.vertices[i % n]
            if point.lon > min(p1_lon, p2_lon):
                if point.lon <= max(p1_lon, p2_lon):
                    if point.lat <= max(p1_lat, p2_lat):
                        if p1_lon != p2_lon:
                            lat_inters = (point.lon - p1_lon) * (p2_lat - p1_lat) / (p2_lon - p1_lon) + p1_lat
                        if p1_lat == p2_lat or point.lat <= lat_inters:
                            inside = not inside
            p1_lat, p1_lon = p2_lat, p2_lon
        return inside

    def intersects_segment(self, seg_start: Tuple[float, float], seg_end: Tuple[float, float]) -> bool:
        """Tests if line segment intersects polygon bounding edges or has endpoints inside."""
        p_start = Point(seg_start[0], seg_start[1])
        p_end = Point(seg_end[0], seg_end[1])
        if self.contains_point(p_start) or self.contains_point(p_end):
            return True

        def ccw(A, B, C):
            return (C[1]-A[1]) * (B[0]-A[0]) > (B[1]-A[1]) * (C[0]-A[0])

        def intersect(A, B, C, D):
            return ccw(A, C, D) != ccw(B, C, D) and ccw(A, B, C) != ccw(A, B, D)

        n = len(self.vertices)
        for i in range(n):
            v1 = self.vertices[i]
            v2 = self.vertices[(i + 1) % n]
            if intersect(seg_start, seg_end, v1, v2):
                return True
        return False


class GISImpactValidator:
    """
    Executes geometric unit testing of the spatial impact assessment engine.
    """

    @classmethod
    def run_deterministic_tests(cls) -> Dict[str, Any]:
        # Defined storm polygon: A box over [28.5 to 28.7 N, 77.1 to 77.3 E]
        storm_poly = Polygon([
            (28.50, 77.10),
            (28.70, 77.10),
            (28.70, 77.30),
            (28.50, 77.30)
        ])

        # Test Points: Schools
        schools = [
            Point(28.60, 77.20, name="Central High School", metadata={"type": "school"}),  # INSIDE
            Point(28.65, 77.25, name="Model Academy", metadata={"type": "school"}),       # INSIDE
            Point(28.80, 77.40, name="North Secondary", metadata={"type": "school"}),      # OUTSIDE
            Point(28.40, 77.05, name="South Elementary", metadata={"type": "school"})      # OUTSIDE
        ]
        impacted_schools = [s.name for s in schools if storm_poly.contains_point(s)]
        schools_expected = ["Central High School", "Model Academy"]
        schools_passed = (sorted(impacted_schools) == sorted(schools_expected))

        # Test Points: Hospitals
        hospitals = [
            Point(28.58, 77.15, name="City Civil Hospital", metadata={"beds": 250}),      # INSIDE
            Point(28.90, 77.50, name="District Trauma Center", metadata={"beds": 180})    # OUTSIDE
        ]
        impacted_hospitals = [h.name for h in hospitals if storm_poly.contains_point(h)]
        hospitals_expected = ["City Civil Hospital"]
        hospitals_passed = (impacted_hospitals == hospitals_expected)

        # Test Lines: Highways
        highways = [
            ("NH-44 Corridor", (28.45, 77.20), (28.75, 77.20)),  # CUTS THROUGH POLYGON
            ("SH-12 Bypass", (28.85, 77.35), (28.95, 77.45))     # COMPLETELY OUTSIDE
        ]
        impacted_highways = [name for name, p1, p2 in highways if storm_poly.intersects_segment(p1, p2)]
        highways_expected = ["NH-44 Corridor"]
        highways_passed = (impacted_highways == highways_expected)

        # Test Population Grid (mock 4 cells)
        pop_cells = [
            ((28.55, 77.15), 12500),  # inside
            ((28.65, 77.25), 18200),  # inside
            ((28.85, 77.45), 24000)   # outside
        ]
        exposed_population = sum(pop for (lat, lon), pop in pop_cells if storm_poly.contains_point(Point(lat, lon)))
        pop_expected = 12500 + 18200
        pop_passed = (exposed_population == pop_expected)

        all_tests_passed = schools_passed and hospitals_passed and highways_passed and pop_passed

        return {
            "validation_type": "Deterministic Software Verification (GIS Geometry Intersection)",
            "disclaimer": "This is software validation, not disaster prediction validation.",
            "overall_status": "PASSED" if all_tests_passed else "FAILED",
            "test_summary": {
                "school_layer_intersection": {
                    "passed": schools_passed,
                    "detected": impacted_schools,
                    "expected": schools_expected
                },
                "hospital_layer_intersection": {
                    "passed": hospitals_passed,
                    "detected": impacted_hospitals,
                    "expected": hospitals_expected
                },
                "highway_corridor_intersection": {
                    "passed": highways_passed,
                    "detected": impacted_highways,
                    "expected": highways_expected
                },
                "exposed_population_aggregation": {
                    "passed": pop_passed,
                    "calculated_population": exposed_population,
                    "expected_population": pop_expected
                }
            }
        }
