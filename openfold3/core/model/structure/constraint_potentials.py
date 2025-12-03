# Copyright 2025 AlQuraishi Laboratory
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Constraint potentials for enforcing minimum distance separation between domains.

This module implements Boltz-style steering potentials for diffusion-based
structure prediction, specifically designed for membrane proteins where
extracellular, transmembrane, and intracellular domains must maintain
minimum separation distances.

Reference: Boltz-1 (https://github.com/jwohlwend/boltz)
"""

import torch
import torch.nn.functional as F


def compute_domain_centroids(
    positions: torch.Tensor,
    domain_masks: torch.Tensor,
    atom_mask: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Compute the centroid (center of mass) for each domain.

    Args:
        positions:
            [*, N_atom, 3] Atom positions
        domain_masks:
            [*, N_atom, num_domains] Binary masks indicating which atoms
            belong to each domain
        atom_mask:
            [*, N_atom] Atom mask indicating valid atoms
        eps:
            Small constant for numerical stability

    Returns:
        [*, num_domains, 3] Centroid positions for each domain
    """
    # Apply atom mask to domain masks
    # [*, N_atom, num_domains]
    valid_domain_masks = domain_masks * atom_mask[..., None]

    # Compute weighted sum of positions for each domain
    # [*, num_domains, 3]
    weighted_positions = torch.einsum(
        "...ad,...ac->...dc", valid_domain_masks, positions
    )

    # Count atoms per domain for normalization
    # [*, num_domains]
    atom_counts = valid_domain_masks.sum(dim=-2) + eps

    # Compute centroids
    # [*, num_domains, 3]
    centroids = weighted_positions / atom_counts[..., None]

    return centroids


def time_dependent_threshold(
    t: torch.Tensor,
    t_max: float,
    d_max: float = 5.0,
    d_min: float = 1.0,
) -> torch.Tensor:
    """
    Compute time-dependent minimum distance threshold using Boltz scheduling.

    The threshold smoothly interpolates from d_max at t=t_max (high noise)
    to d_min at t=0 (final structure). This allows domains to separate
    early in the diffusion process.

    Formula: b_t = d_min + (d_max - d_min) * exp_factor
    where exp_factor = (exp(-2(1-t_norm)) - 1) / (exp(-2) - 1)

    Args:
        t:
            [*] Current noise level
        t_max:
            Maximum noise level (from noise schedule)
        d_max:
            Maximum threshold at t=t_max (default 5.0 Å, per Boltz)
        d_min:
            Minimum threshold at t=0 (default 1.0 Å, per Boltz)

    Returns:
        [*] Time-dependent threshold value
    """
    # Normalize t to [0, 1]
    t_normalized = t / t_max

    # Exponential interpolation (Boltz formula)
    exp_neg2 = torch.exp(torch.tensor(-2.0, device=t.device, dtype=t.dtype))
    exp_factor = (torch.exp(-2.0 * (1.0 - t_normalized)) - 1.0) / (exp_neg2 - 1.0)

    return d_min + (d_max - d_min) * exp_factor


def domain_separation_potential(
    positions: torch.Tensor,
    domain_masks: torch.Tensor,
    atom_mask: torch.Tensor,
    min_distances: torch.Tensor,
    t: torch.Tensor,
    t_max: float,
    scale_with_time: bool = True,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Compute flat-bottom potential for domain separation constraints.

    This implements a Boltz-style potential where:
    - E = 0 when constraint is satisfied (domains are far enough apart)
    - E increases quadratically as domains get too close

    Args:
        positions:
            [*, N_atom, 3] Atom positions
        domain_masks:
            [*, N_atom, num_domains] Binary masks for each domain
        atom_mask:
            [*, N_atom] Valid atom mask
        min_distances:
            [num_domains, num_domains] Minimum required distances between
            domain centroids (in Angstroms). Only upper triangle is used.
        t:
            [*] Current noise level
        t_max:
            Maximum noise level
        scale_with_time:
            If True, scale min_distances by time-dependent factor
        eps:
            Small constant for numerical stability

    Returns:
        [*] Energy penalty for constraint violations
    """
    # Compute domain centroids
    # [*, num_domains, 3]
    centroids = compute_domain_centroids(positions, domain_masks, atom_mask, eps)

    # Compute pairwise distances between centroids
    # [*, num_domains, num_domains]
    dists = torch.cdist(centroids, centroids, p=2)

    # Apply time-dependent scaling if requested
    if scale_with_time:
        b_t = time_dependent_threshold(t, t_max)
        # Scale the minimum distances
        # For early diffusion (high t), we want larger separation
        scaled_min_distances = min_distances * b_t[..., None, None]
    else:
        scaled_min_distances = min_distances

    # Flat-bottom potential: penalize if distance < min_distance
    # violations = max(min_distance - actual_distance, 0)
    violations = F.relu(scaled_min_distances - dists)

    # Only consider upper triangle (avoid double counting)
    num_domains = domain_masks.shape[-1]
    triu_mask = torch.triu(
        torch.ones(num_domains, num_domains, device=positions.device, dtype=positions.dtype),
        diagonal=1
    )

    # Quadratic penalty
    energy = (violations ** 2 * triu_mask).sum(dim=(-1, -2))

    return energy


def compute_separation_gradient(
    positions: torch.Tensor,
    domain_masks: torch.Tensor,
    atom_mask: torch.Tensor,
    min_distances: torch.Tensor,
    t: torch.Tensor,
    t_max: float,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Compute gradient of separation potential with respect to positions.

    This is used for gradient-based constraint enforcement during diffusion.

    Args:
        positions:
            [*, N_atom, 3] Atom positions (will be modified to require grad)
        domain_masks:
            [*, N_atom, num_domains] Binary masks for each domain
        atom_mask:
            [*, N_atom] Valid atom mask
        min_distances:
            [num_domains, num_domains] Minimum required distances
        t:
            [*] Current noise level
        t_max:
            Maximum noise level
        eps:
            Small constant for numerical stability

    Returns:
        [*, N_atom, 3] Gradient of energy with respect to positions
    """
    # Enable gradients for positions
    positions_grad = positions.clone().requires_grad_(True)

    # Compute energy
    energy = domain_separation_potential(
        positions_grad,
        domain_masks,
        atom_mask,
        min_distances,
        t,
        t_max,
        scale_with_time=True,
        eps=eps,
    )

    # Compute gradient
    if energy.sum() > 0:
        grad = torch.autograd.grad(
            energy.sum(),
            positions_grad,
            create_graph=False,
            retain_graph=False,
        )[0]
    else:
        grad = torch.zeros_like(positions)

    return grad


def apply_separation_constraints(
    positions: torch.Tensor,
    batch: dict,
    t: torch.Tensor,
    t_max: float,
    gradient_scale: float = 0.1,
    num_steps: int = 1,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Apply domain separation constraints to predicted positions.

    This function enforces minimum distance constraints between domains
    by iteratively pushing domains apart using gradient descent on the
    separation potential.

    Args:
        positions:
            [*, N_atom, 3] Predicted atom positions
        batch:
            Feature dictionary containing:
            - "atom_mask": [*, N_atom] Valid atom mask
            - "domain_masks": [*, N_atom, num_domains] Domain membership masks
            - "domain_min_distances": [num_domains, num_domains] Min distances
        t:
            [*] Current noise level
        t_max:
            Maximum noise level (first element of noise schedule)
        gradient_scale:
            Step size for gradient descent (default 0.1)
        num_steps:
            Number of gradient descent steps (default 1)
        eps:
            Small constant for numerical stability

    Returns:
        [*, N_atom, 3] Constrained atom positions
    """
    # Check if constraints are specified
    domain_masks = batch.get("domain_masks")
    min_distances = batch.get("domain_min_distances")

    if domain_masks is None or min_distances is None:
        return positions

    atom_mask = batch["atom_mask"]

    # Compute initial energy for logging
    initial_energy = domain_separation_potential(
        positions, domain_masks, atom_mask, min_distances, t, t_max, eps=eps
    )

    # Iteratively apply gradient descent
    constrained_positions = positions.clone()

    for step in range(num_steps):
        # Compute gradient of separation potential
        grad = compute_separation_gradient(
            constrained_positions,
            domain_masks,
            atom_mask,
            min_distances,
            t,
            t_max,
            eps=eps,
        )

        # Update positions (gradient descent to minimize energy)
        constrained_positions = constrained_positions - gradient_scale * grad

        # Ensure masked atoms stay at zero
        constrained_positions = constrained_positions * atom_mask[..., None]

    # Compute final energy and log if there was a violation
    final_energy = domain_separation_potential(
        constrained_positions, domain_masks, atom_mask, min_distances, t, t_max, eps=eps
    )

    if initial_energy.sum() > 0:
        print(
            f"[Constraint] t={t.mean().item():.4f} "
            f"energy: {initial_energy.mean().item():.4f} -> {final_energy.mean().item():.4f} "
            f"grad_norm={grad.norm().item():.4f}",
            flush=True
        )

    return constrained_positions


def create_domain_masks_from_residue_ranges(
    num_atoms: int,
    atom_to_token: torch.Tensor,
    domain_ranges: dict[str, tuple[int, int]],
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, list[str]]:
    """
    Create domain masks from residue range specifications.

    Args:
        num_atoms:
            Total number of atoms
        atom_to_token:
            [N_atom] Mapping from atom index to token/residue index
        domain_ranges:
            Dictionary mapping domain names to (start, end) residue ranges
            e.g., {"extracellular": (28, 906), "transmembrane": (930, 952)}
        device:
            Device for tensors
        dtype:
            Data type for tensors

    Returns:
        domain_masks: [N_atom, num_domains] Binary masks
        domain_names: List of domain names in order
    """
    domain_names = list(domain_ranges.keys())
    num_domains = len(domain_names)

    domain_masks = torch.zeros(num_atoms, num_domains, device=device, dtype=dtype)

    for i, (name, (start, end)) in enumerate(domain_ranges.items()):
        # Mark atoms belonging to residues in this range
        in_range = (atom_to_token >= start) & (atom_to_token <= end)
        domain_masks[:, i] = in_range.to(dtype)

    return domain_masks, domain_names


def create_min_distance_matrix(
    domain_names: list[str],
    distance_specs: dict[tuple[str, str], float],
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    Create minimum distance matrix from pairwise specifications.

    Args:
        domain_names:
            List of domain names
        distance_specs:
            Dictionary mapping (domain1, domain2) pairs to minimum distances
            e.g., {("extracellular", "intracellular"): 30.0}
        device:
            Device for tensor
        dtype:
            Data type for tensor

    Returns:
        [num_domains, num_domains] Symmetric matrix of minimum distances
    """
    num_domains = len(domain_names)
    name_to_idx = {name: i for i, name in enumerate(domain_names)}

    min_distances = torch.zeros(num_domains, num_domains, device=device, dtype=dtype)

    for (d1, d2), dist in distance_specs.items():
        if d1 in name_to_idx and d2 in name_to_idx:
            i, j = name_to_idx[d1], name_to_idx[d2]
            min_distances[i, j] = dist
            min_distances[j, i] = dist  # Symmetric

    return min_distances
