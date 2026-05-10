"""Review tracking issue creation for MalaOrchestrator.

This module handles creating beads issues from low-priority (P2/P3)
review findings that didn't block the review but should be tracked.
"""

from __future__ import annotations

import hashlib
import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.core.protocols.events import MalaEventSink
    from src.core.protocols.issue import IssueProvider
    from src.core.protocols.review import ReviewIssueProtocol


def _get_finding_fingerprint(issue: ReviewIssueProtocol) -> str:
    """Generate a unique fingerprint for a single finding.

    Returns a hex hash to ensure safe regex matching (no special characters).
    """
    content = f"{issue.file}:{issue.line_start}:{issue.line_end}:{issue.title}"
    return hashlib.sha256(content.encode()).hexdigest()[:16]


def _build_findings_section(
    review_issues: list[ReviewIssueProtocol],
    start_idx: int = 1,
) -> tuple[str, str, list[str]]:
    """Build markdown sections for review findings.

    Args:
        review_issues: List of review issues to format.
        start_idx: Starting index for finding numbering.

    Returns:
        Tuple of (formatted sections string, batch dedup tag, list of individual fingerprints).
    """
    # Build fingerprints for each finding
    finding_fingerprints = [_get_finding_fingerprint(issue) for issue in review_issues]
    sorted_fingerprints = sorted(finding_fingerprints)
    content_hash = hashlib.sha256("|".join(sorted_fingerprints).encode()).hexdigest()[
        :12
    ]
    dedup_tag = f"review_finding:{content_hash}"

    parts: list[str] = []
    for idx, issue in enumerate(review_issues, start_idx):
        file_path = issue.file
        line_start = issue.line_start
        line_end = issue.line_end
        priority = issue.priority
        title = issue.title
        body = issue.body
        reviewer = issue.reviewer

        finding_priority = f"P{priority}" if priority is not None else "P3"

        # Build location string
        if line_start == line_end or line_end == 0:
            location = f"{file_path}:{line_start}" if file_path else ""
        else:
            location = f"{file_path}:{line_start}-{line_end}" if file_path else ""

        parts.append(f"### Finding {idx}: {title}")
        parts.append("")
        parts.append(f"**Priority:** {finding_priority}")
        parts.append(f"**Reviewer:** {reviewer}")
        if location:
            parts.append(f"**Location:** {location}")
        if body:
            parts.extend(["", body])
        parts.extend(["", "---", ""])

    return "\n".join(parts), dedup_tag, finding_fingerprints


def _extract_existing_fingerprints(description: str) -> set[str]:
    """Extract individual finding fingerprints from existing description.

    Fingerprints are stored as HTML comments: <!-- fp:hex_hash -->
    We use a hex hash to avoid issues with special characters in titles.

    Also supports legacy format <!-- fp:file:line:line:title --> for backwards
    compatibility with existing tracking issues. Legacy fingerprints are hashed
    to match the format used by _get_finding_fingerprint.
    """
    # Match new hex-only format (16 hex chars)
    hex_pattern = r"<!-- fp:([a-f0-9]{16}) -->"
    hex_matches = set(re.findall(hex_pattern, description))

    # Match legacy format (file:line:line:title) for backwards compatibility
    # Legacy fingerprints contain colons and non-hex characters
    legacy_pattern = r"<!-- fp:([^>]+:[^>]+) -->"
    legacy_matches = re.findall(legacy_pattern, description)
    # Hash legacy fingerprints to match the format used by _get_finding_fingerprint
    legacy_hashes = {
        hashlib.sha256(m.encode()).hexdigest()[:16] for m in legacy_matches
    }

    return hex_matches | legacy_hashes


def _update_header_count(description: str, new_count: int) -> str:
    """Update the finding count in the description header using regex.

    Handles both singular and plural forms. Targets the specific header pattern
    to avoid matching similar text in finding bodies.
    """
    plural_s = "s" if new_count != 1 else ""
    # Match specifically "consolidates N non-blocking finding(s)" to avoid false matches
    pattern = r"consolidates \d+ non-blocking findings?"
    replacement = f"consolidates {new_count} non-blocking finding{plural_s}"
    return re.sub(pattern, replacement, description)


async def create_review_tracking_issues(
    beads: IssueProvider,
    event_sink: MalaEventSink,
    source_issue_id: str,
    review_issues: list[ReviewIssueProtocol],
    parent_epic_id: str | None = None,
) -> None:
    """Create or update a beads issue from P2/P3 review findings.

    All low-priority issues that didn't block the review are consolidated
    into a single tracking issue per source issue. If a tracking issue already
    exists for this source, new findings are appended to it.

    Args:
        beads: Issue provider for creating/updating issues.
        event_sink: Event sink for warnings.
        source_issue_id: The issue ID that triggered the review.
        review_issues: List of ReviewIssueProtocol objects from the review.
        parent_epic_id: Optional parent epic ID to attach new tracking issues to.
    """
    if not review_issues:
        return

    # Build the new findings section and get a content-based dedup tag
    new_findings_section, new_dedup_tag, new_fingerprints = _build_findings_section(
        review_issues
    )

    # Check for existing tracking issue for this source
    source_tag = f"source:{source_issue_id}"
    existing_id = await beads.find_issue_by_tag_async(source_tag)

    if existing_id:
        # Fetch existing description - skip update on failure (Finding 4)
        existing_desc = await beads.get_issue_description_async(existing_id)
        if existing_desc is None:
            event_sink.on_warning(
                f"Failed to fetch description for {existing_id}, skipping update",
                agent_id=source_issue_id,
            )
            return

        # Check batch-level dedup first (fast path)
        if new_dedup_tag in existing_desc:
            return

        # Finding 6: Filter out individually duplicate findings
        existing_fingerprints = _extract_existing_fingerprints(existing_desc)
        unique_issues = [
            issue
            for issue in review_issues
            if _get_finding_fingerprint(issue) not in existing_fingerprints
        ]

        if not unique_issues:
            # All findings already exist individually
            return

        # Append new findings to existing issue
        # Count existing findings to continue numbering
        existing_finding_count = existing_desc.count("### Finding ")
        new_findings_section, new_dedup_tag, unique_fingerprints = (
            _build_findings_section(unique_issues, start_idx=existing_finding_count + 1)
        )

        # Build updated description with proper count (Findings 2, 5)
        total_count = existing_finding_count + len(unique_issues)
        updated_desc = _update_header_count(existing_desc, total_count)

        # Add fingerprint markers for individual dedup (Finding 6)
        fingerprint_comments = "\n".join(
            f"<!-- fp:{fp} -->" for fp in unique_fingerprints
        )

        # Append new findings and dedup tag before the end
        updated_desc = (
            updated_desc.rstrip()
            + f"\n\n{new_findings_section}\n{fingerprint_comments}\n<!-- {new_dedup_tag} -->\n"
        )

        # Finding 3: Compute new highest priority across all findings
        new_priorities = [i.priority for i in unique_issues if i.priority is not None]
        new_highest = min(new_priorities) if new_priorities else 3

        # Extract current highest priority from description
        priority_match = re.search(r"\*\*Highest priority:\*\* P(\d+)", existing_desc)
        current_highest = int(priority_match.group(1)) if priority_match else 3

        # Update if new findings have higher priority (lower number)
        final_highest = min(current_highest, new_highest)
        if final_highest != current_highest:
            updated_desc = re.sub(
                r"\*\*Highest priority:\*\* P\d+",
                f"**Highest priority:** P{final_highest}",
                updated_desc,
            )

        # Finding 3: Update issue title
        plural_s = "s" if total_count != 1 else ""
        new_title = f"[Review] {total_count} non-blocking finding{plural_s} from {source_issue_id}"

        # Finding 1: Check return value of update
        update_success = await beads.update_issue_description_async(
            existing_id, updated_desc
        )
        if not update_success:
            event_sink.on_warning(
                f"Failed to update tracking issue {existing_id}",
                agent_id=source_issue_id,
            )
            return

        # Update title and priority (Finding 3)
        title_update_success = await beads.update_issue_async(
            existing_id,
            title=new_title,
            priority=f"P{final_highest}",
        )
        if not title_update_success:
            event_sink.on_warning(
                f"Failed to update title/priority for tracking issue {existing_id}",
                agent_id=source_issue_id,
            )

        event_sink.on_warning(
            f"Appended {len(unique_issues)} finding{'s' if len(unique_issues) > 1 else ''} to tracking issue {existing_id}",
            agent_id=source_issue_id,
        )
        return

    # No existing issue - create a new one
    # Determine highest priority among findings (lowest number = highest priority)
    priorities = [i.priority for i in review_issues if i.priority is not None]
    highest_priority = min(priorities) if priorities else 3
    priority_str = f"P{highest_priority}"

    # Build consolidated issue title
    issue_count = len(review_issues)
    issue_title = f"[Review] {issue_count} non-blocking finding{'s' if issue_count > 1 else ''} from {source_issue_id}"

    # Add fingerprint markers for individual dedup (Finding 6)
    fingerprint_comments = "\n".join(f"<!-- fp:{fp} -->" for fp in new_fingerprints)

    # Build description with all findings
    description_parts = [
        "## Review Findings",
        "",
        f"This issue consolidates {issue_count} non-blocking finding{'s' if issue_count > 1 else ''} from code review.",
        "",
        f"**Source issue:** {source_issue_id}",
        f"**Highest priority:** {priority_str}",
        "",
        "---",
        "",
        new_findings_section,
        fingerprint_comments,
        f"<!-- {new_dedup_tag} -->",
    ]

    description = "\n".join(description_parts)

    # Tags for tracking
    tags = [
        "auto_generated",
        "review_finding",
        source_tag,
    ]

    new_issue_id = await beads.create_issue_async(
        title=issue_title,
        description=description,
        priority=priority_str,
        tags=tags,
        parent_id=parent_epic_id,
    )
    if new_issue_id:
        event_sink.on_warning(
            f"Created tracking issue {new_issue_id} for {issue_count} {priority_str}+ review finding{'s' if issue_count > 1 else ''}",
            agent_id=source_issue_id,
        )
