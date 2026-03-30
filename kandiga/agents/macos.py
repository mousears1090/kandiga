"""macOS native integrations via osascript/AppleScript.

These register as tools in the agent pipeline, giving the model
direct access to macOS apps — something cloud agents can't do.
"""

from __future__ import annotations

import json
import subprocess
from typing import Optional

from kandiga.agents.tools import ToolRegistry


def _run_osascript(script: str, timeout: int = 5) -> str:
    """Execute AppleScript and return the result."""
    try:
        result = subprocess.run(
            ["osascript", "-e", script],
            capture_output=True, text=True, timeout=timeout,
        )
        if result.returncode != 0:
            err = result.stderr.strip()
            if "not running" in err.lower():
                return f"Error: app not running"
            return f"Error: {err}"
        return result.stdout.strip() or "(no results)"
    except subprocess.TimeoutExpired:
        return "Error: timed out (app may not be running)"
    except FileNotFoundError:
        return "Error: osascript not found (not macOS?)"
    except Exception as e:
        return f"Error: {e}"


# --- Calendar ---

def calendar_list_events(days: int = 7) -> str:
    """List upcoming calendar events for the next N days."""
    script = f'''
    set output to ""
    tell application "Calendar"
        set startDate to current date
        set endDate to startDate + ({days} * days)
        repeat with cal in calendars
            set evts to (every event of cal whose start date >= startDate and start date <= endDate)
            repeat with evt in evts
                set output to output & (summary of evt) & " | " & (start date of evt as string) & " | " & (name of cal) & linefeed
            end repeat
        end repeat
    end tell
    return output
    '''
    return _run_osascript(script) or "(no events)"


def calendar_create_event(title: str, date: str, time: str = "09:00", duration_minutes: int = 60) -> str:
    """Create a calendar event. Date format: YYYY-MM-DD, time: HH:MM."""
    script = f'''
    tell application "Calendar"
        tell calendar 1
            set startDate to date "{date} {time}"
            set endDate to startDate + ({duration_minutes} * minutes)
            make new event with properties {{summary:"{title}", start date:startDate, end date:endDate}}
        end tell
    end tell
    return "Created event: {title} on {date} at {time}"
    '''
    return _run_osascript(script)


# --- Reminders ---

def reminders_list() -> str:
    """List all incomplete reminders."""
    script = '''
    set output to ""
    tell application "Reminders"
        repeat with r in (every reminder whose completed is false)
            set output to output & (name of r)
            if due date of r is not missing value then
                set output to output & " | due: " & (due date of r as string)
            end if
            set output to output & linefeed
        end repeat
    end tell
    return output
    '''
    return _run_osascript(script) or "(no reminders)"


def reminders_create(title: str, notes: str = "") -> str:
    """Create a new reminder."""
    notes_prop = f', body:"{notes}"' if notes else ""
    script = f'''
    tell application "Reminders"
        tell default list
            make new reminder with properties {{name:"{title}"{notes_prop}}}
        end tell
    end tell
    return "Created reminder: {title}"
    '''
    return _run_osascript(script)


# --- Notes ---

def notes_search(query: str) -> str:
    """Search Apple Notes for a keyword."""
    script = f'''
    set output to ""
    tell application "Notes"
        set matchingNotes to every note whose name contains "{query}" or body contains "{query}"
        repeat with n in matchingNotes
            set output to output & (name of n) & " | " & (modification date of n as string) & linefeed
        end repeat
    end tell
    return output
    '''
    return _run_osascript(script) or f"(no notes matching '{query}')"


def notes_create(title: str, body: str) -> str:
    """Create a new Apple Note."""
    # Escape for AppleScript
    body_escaped = body.replace('"', '\\"').replace('\n', '\\n')
    script = f'''
    tell application "Notes"
        tell default account
            make new note at folder "Notes" with properties {{name:"{title}", body:"{body_escaped}"}}
        end tell
    end tell
    return "Created note: {title}"
    '''
    return _run_osascript(script)


# --- Notifications ---

def notify(title: str, message: str) -> str:
    """Show a macOS notification."""
    script = f'display notification "{message}" with title "{title}"'
    return _run_osascript(script)


# --- Finder ---

def finder_reveal(path: str) -> str:
    """Reveal a file/folder in Finder."""
    script = f'''
    tell application "Finder"
        reveal POSIX file "{path}"
        activate
    end tell
    return "Revealed in Finder: {path}"
    '''
    return _run_osascript(script)


# --- Contacts ---

def contacts_search(name: str) -> str:
    """Search contacts by name."""
    script = f'''
    set output to ""
    tell application "Contacts"
        set matches to every person whose name contains "{name}"
        repeat with p in matches
            set output to output & (name of p)
            try
                set output to output & " | " & (value of first email of p)
            end try
            try
                set output to output & " | " & (value of first phone of p)
            end try
            set output to output & linefeed
        end repeat
    end tell
    return output
    '''
    return _run_osascript(script) or f"(no contacts matching '{name}')"


# --- Register all macOS tools ---

def system_info() -> str:
    """Get basic system info."""
    try:
        r = subprocess.run(
            ["system_profiler", "SPHardwareDataType", "-detailLevel", "mini"],
            capture_output=True, text=True, timeout=10,
        )
        return r.stdout[:2000] if r.stdout else "Error: no output"
    except Exception as e:
        return f"Error: {e}"


def say(text: str) -> str:
    """Speak text aloud using macOS text-to-speech."""
    try:
        subprocess.Popen(["say", text])
        return f"Speaking: {text}"
    except Exception as e:
        return f"Error: {e}"


def register_macos_tools(registry: ToolRegistry) -> int:
    """Register all macOS tools into a ToolRegistry. Returns count."""
    tools = [
        ("calendar_list", "List upcoming calendar events", {"days": "int"}, calendar_list_events),
        ("calendar_create", "Create a calendar event", {"title": "str", "date": "str", "time": "str", "duration_minutes": "int"}, calendar_create_event),
        ("reminders_list", "List incomplete reminders", {}, reminders_list),
        ("reminders_create", "Create a reminder", {"title": "str", "notes": "str"}, reminders_create),
        ("notes_search", "Search Apple Notes", {"query": "str"}, notes_search),
        ("notes_create", "Create an Apple Note", {"title": "str", "body": "str"}, notes_create),
        ("notify", "Show a macOS notification", {"title": "str", "message": "str"}, notify),
        ("finder_reveal", "Reveal a file in Finder", {"path": "str"}, finder_reveal),
        ("contacts_search", "Search contacts", {"name": "str"}, contacts_search),
        ("system_info", "Get Mac hardware info", {}, system_info),
        ("say", "Speak text aloud", {"text": "str"}, say),
    ]
    for name, desc, schema, func in tools:
        registry.register(name, desc, schema, func)
    return len(tools)
