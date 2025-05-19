import pytest
import io
from PIL import Image
from app import extract_gps_from_image

# Mock EXIF data
class MockTag:
    def __init__(self, values):
        self.values = values

class MockRatio:
    def __init__(self, num, den):
        self.num = num
        self.den = den

def test_extract_gps_from_image():
    # Δημιουργία mock εικόνας με EXIF δεδομένα
    class MockFile:
        def __init__(self):
            self.seek = lambda x: None
    mock_file = MockFile()
    
    # Mock EXIF tags
    mock_tags = {
        "GPS GPSLatitudeRef": MockTag("N"),
        "GPS GPSLongitudeRef": MockTag("E"),
        "GPS GPSLatitude": MockTag([MockRatio(40, 1), MockRatio(30, 1), MockRatio(0, 1)]),
        "GPS GPSLongitude": MockTag([MockRatio(73, 1), MockRatio(45, 1), MockRatio(0, 1)])
    }
    
    # Monkey patch exifread
    import exifread
    original_process_file = exifread.process_file
    exifread.process_file = lambda *args, **kwargs: mock_tags
    
    lat, lon = extract_gps_from_image(mock_file)
    assert abs(lat - 40.5) < 0.01, "Latitude calculation incorrect"
    assert abs(lon - 73.75) < 0.01, "Longitude calculation incorrect"
    
    # Επαναφορά original function
    exifread.process_file = original_process_file

def test_extract_gps_no_data():
    class MockFile:
        def __init__(self):
            self.seek = lambda x: None
    mock_file = MockFile()
    
    # Mock EXIF χωρίς GPS
    import exifread
    original_process_file = exifread.process_file
    exifread.process_file = lambda *args, **kwargs: {}
    
    lat, lon = extract_gps_from_image(mock_file)
    assert lat is None
    assert lon is None
    
    exifread.process_file = original_process_file
