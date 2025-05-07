#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Simulation Classes

This module contains the class definitions for the simulation framework.

Author: Your Name
Date: 2025-03-31
"""

from typing import Dict, List, Tuple, Optional

class Cell:
    """
    Represents a cell in the simulation with basic metadata and mutation profiles.
    """
    __slots__ = [
        'id', 'parent_id', 'generation', 'time',
        'mutation_afs', 'cell_type', 'children', 'mutation_profile'
    ]
    
    def __init__(
        self,
        cid: int,
        parent_id: Optional[int],
        generation: int,
        time_point: float,
        mutation_afs: Optional[Dict[str, float]] = None,
        cell_type: str = 'StemCell'
    ) -> None:
        """
        Initialize a Cell object.

        Parameters
        ----------
        cid : int
            Numerical ID for this cell.
        parent_id : int or None
            ID of the parent cell (None if this is the root).
        generation : int
            Generation of the cell (parent generation + 1).
        time_point : float
            Time at which this cell is considered to exist.
        mutation_afs : dict, optional
            Maps mutation ID -> allele frequency (default None).
        cell_type : str
            The cell type (e.g., 'StemCell', 'Progenitor1', etc.).
        """
        self.id = f"cell_{cid}"
        self.parent_id = None if parent_id is None else f"cell_{parent_id}"
        self.generation = generation
        self.time = time_point
        self.mutation_afs = mutation_afs if mutation_afs else {}
        self.cell_type = cell_type
        self.children: List[str] = []
        self.mutation_profile: Dict[str, Dict[str, float]] = {}  # e.g., { mut_id: {'DP':..., 'AD':..., ...} }