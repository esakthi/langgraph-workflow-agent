import unittest
from unittest.mock import patch
from app import create_event_tool

class TestToolCalling(unittest.TestCase):

    @patch("app.create_google_calendar_event")
    def test_create_event_tool(self, mock_create_event):
        """
        Tests the create_event_tool function to ensure it correctly formats
        the event details and calls the Google Calendar API function.
        """
        # Define sample inputs for the tool
        summary = "Team Meeting"
        start_datetime = "2024-05-21T14:00:00"
        end_datetime = "2024-05-21T15:00:00"

        # Call the tool function
        create_event_tool(summary, start_datetime, end_datetime)

        # Define the expected event details that should be passed to the mock
        expected_event_details = {
            "summary": summary,
            "start": {"dateTime": start_datetime, "timeZone": "America/Los_Angeles"},
            "end": {"dateTime": end_datetime, "timeZone": "America/Los_Angeles"},
        }

        # Assert that the create_google_calendar_event function was called once
        # with the correctly formatted event details.
        mock_create_event.assert_called_once_with(expected_event_details)

if __name__ == "__main__":
    unittest.main()