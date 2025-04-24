from datetime import datetime, timedelta

class CalendarEvent:
    def __init__(self, title, start_time, end_time):
        self.title = title
        self.start_time = start_time
        self.end_time = end_time

    def __repr__(self):
        return f"{self.title}: {self.start_time} - {self.end_time}"

class DummyCalendar:
    def __init__(self):
        self.events = []

    def add_event(self, title, start_time, end_time):
        if self._has_conflict(start_time, end_time):
            return "Time slot not available."
        event = CalendarEvent(title, start_time, end_time)
        self.events.append(event)
        return f"Event '{title}' added."

    def update_event(self, title, new_start_time, new_end_time):
        for event in self.events:
            if event.title == title:
                if self._has_conflict(new_start_time, new_end_time, exclude_title=title):
                    return "New time slot conflicts with another event."
                event.start_time = new_start_time
                event.end_time = new_end_time
                return f"Event '{title}' updated."
        return "Event not found."

    def delete_event(self, title):
        for i, event in enumerate(self.events):
            if event.title == title:
                del self.events[i]
                return f"Event '{title}' deleted."
        return "Event not found."

    def suggest_alternate_time(self, duration_minutes=60):
        now = datetime.now()
        future = now + timedelta(days=1)
        slot = now
        while slot + timedelta(minutes=duration_minutes) < future:
            if not self._has_conflict(slot, slot + timedelta(minutes=duration_minutes)):
                return f"Suggested time: {slot} - {slot + timedelta(minutes=duration_minutes)}"
            slot += timedelta(minutes=30)
        return "No available time slot in the next 24 hours."

    def _has_conflict(self, start_time, end_time, exclude_title=None):
        for event in self.events:
            if exclude_title and event.title == exclude_title:
                continue
            if start_time < event.end_time and end_time > event.start_time:
                return True
        return False

    def list_events(self):
        return sorted(self.events, key=lambda x: x.start_time)

# Example usage
if __name__ == "__main__":
    cal = DummyCalendar()
    now = datetime.now()
    print(cal.add_event("Meeting", now + timedelta(hours=1), now + timedelta(hours=2)))
    print(cal.add_event("Call", now + timedelta(hours=1, minutes=30), now + timedelta(hours=2, minutes=30)))  # Should conflict
    print(cal.suggest_alternate_time())
    print(cal.update_event("Meeting", now + timedelta(hours=3), now + timedelta(hours=4)))
    print(cal.list_events())
    print(cal.delete_event("Call"))
