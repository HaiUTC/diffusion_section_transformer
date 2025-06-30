"""
Unified JSON Schema for Multimodal Layout Generation Dataset
Implements the exact schema specification from instruction.md
"""

from typing import Dict, Any, Optional, Union
from dataclasses import dataclass
from pathlib import Path
import json


@dataclass
class ScreenshotMetadata:
    """Screenshot metadata schema."""
    path: str
    width: int
    height: int


@dataclass
class HTMLStructureData:
    """HTML structure data schema."""
    type: str = "HTMLObject"
    data: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.data is None:
            self.data = {}


@dataclass
class SectionLayoutData:
    """Section layout data schema."""
    type: str = "SectionLayout"
    data: Dict[str, Any] = None
    props: Dict[str, str] = None
    
    def __post_init__(self):
        if self.data is None:
            self.data = {"structure": {}}
        if self.props is None:
            self.props = {}


@dataclass
class UnifiedExample:
    """
    Unified example schema as specified in instruction.md.
    
    Example format:
    {
      "id": "unique_example_id",
      "screenshot": {
        "path": "screenshot.png",
        "width": 1920,
        "height": 1080
      },
      "structure": {
        "type": "HTMLObject",
        "data": {
          "div.container": {
            "h1.heading": { "text": "Hello World" },
            "p.paragraph": { "text": "This is a paragraph" }
          }
        }
      },
      "layout": {
        "type": "SectionLayout",
        "data": {
          "structure": {
            "section@div.container": {
              "heading@h1.heading": "",
              "paragraph@p.paragraph": ""
            }
          }
        },
        "props": {}
      }
    }
    """
    id: str
    screenshot: ScreenshotMetadata
    structure: HTMLStructureData
    layout: SectionLayoutData


class UnifiedSchemaValidator:
    """Validates examples against the unified schema."""
    
    REQUIRED_FIELDS = ['id', 'screenshot', 'structure', 'layout']
    SCREENSHOT_FIELDS = ['path', 'width', 'height']
    STRUCTURE_FIELDS = ['type', 'data']
    LAYOUT_FIELDS = ['type', 'data', 'props']
    
    @classmethod
    def validate_example(cls, example_data: Dict[str, Any]) -> tuple[bool, list[str]]:
        """
        Validate a single example against the unified schema.
        
        Returns:
            (is_valid, error_messages)
        """
        errors = []
        
        # Check top-level required fields
        for field in cls.REQUIRED_FIELDS:
            if field not in example_data:
                errors.append(f"Missing required field: {field}")
        
        if errors:
            return False, errors
        
        # Validate screenshot metadata
        screenshot = example_data.get('screenshot', {})
        for field in cls.SCREENSHOT_FIELDS:
            if field not in screenshot:
                errors.append(f"Missing screenshot field: {field}")
            elif field in ['width', 'height'] and not isinstance(screenshot[field], int):
                errors.append(f"Screenshot {field} must be integer")
        
        # Validate structure data
        structure = example_data.get('structure', {})
        for field in cls.STRUCTURE_FIELDS:
            if field not in structure:
                errors.append(f"Missing structure field: {field}")
        
        if structure.get('type') != 'HTMLObject':
            errors.append("Structure type must be 'HTMLObject'")
        
        # Validate layout data
        layout = example_data.get('layout', {})
        for field in cls.LAYOUT_FIELDS:
            if field not in layout:
                errors.append(f"Missing layout field: {field}")
        
        if layout.get('type') != 'SectionLayout':
            errors.append("Layout type must be 'SectionLayout'")
        
        # Validate @ concatenation syntax in layout structure
        layout_structure = layout.get('data', {}).get('structure', {})
        cls._validate_concatenation_syntax(layout_structure, errors)
        
        # Validate props syntax
        props = layout.get('props', {})
        cls._validate_props_syntax(props, errors)
        
        return len(errors) == 0, errors
    
    @classmethod
    def _validate_concatenation_syntax(cls, structure: Dict[str, Any], errors: list[str]):
        """Validate @ concatenation syntax in layout structure."""
        for key, value in structure.items():
            if '@' not in key:
                errors.append(f"Layout key must contain @ concatenation: {key}")
                continue
            
            # Check that @ concatenation is properly formatted
            parts = key.split('@')
            if len(parts) < 2:
                errors.append(f"Invalid @ concatenation format: {key}")
            
            # Recursively validate nested structures
            if isinstance(value, dict):
                cls._validate_concatenation_syntax(value, errors)
    
    @classmethod
    def _validate_props_syntax(cls, props: Dict[str, str], errors: list[str]):
        """Validate props syntax (bi, bo, bv)."""
        valid_prop_keys = {'bi', 'bo', 'bv'}  # background image, overlay, video
        
        for prop_key in props.keys():
            if prop_key not in valid_prop_keys:
                errors.append(f"Invalid prop key: {prop_key}. Must be one of {valid_prop_keys}")
    
    @classmethod
    def load_and_validate_example(cls, example_path: Path) -> tuple[Optional[UnifiedExample], list[str]]:
        """
        Load and validate an example from JSON file.
        
        Returns:
            (example_object, error_messages)
        """
        try:
            with open(example_path, 'r') as f:
                data = json.load(f)
            
            is_valid, errors = cls.validate_example(data)
            
            if not is_valid:
                return None, errors
            
            # Create UnifiedExample object
            example = UnifiedExample(
                id=data['id'],
                screenshot=ScreenshotMetadata(**data['screenshot']),
                structure=HTMLStructureData(**data['structure']),
                layout=SectionLayoutData(**data['layout'])
            )
            
            return example, []
            
        except json.JSONDecodeError as e:
            return None, [f"JSON decode error: {e}"]
        except FileNotFoundError:
            return None, [f"File not found: {example_path}"]
        except Exception as e:
            return None, [f"Unexpected error: {e}"]


def create_example_template(example_id: str, screenshot_path: str = "screenshot.png") -> Dict[str, Any]:
    """Create a template for a new example following the unified schema."""
    return {
        "id": example_id,
        "screenshot": {
            "path": screenshot_path,
            "width": 1920,
            "height": 1080
        },
        "structure": {
            "type": "HTMLObject",
            "data": {
                "div.container": {
                    "h1.heading": {"text": "Hello World"},
                    "p.paragraph": {"text": "This is a paragraph"}
                }
            }
        },
        "layout": {
            "type": "SectionLayout",
            "data": {
                "structure": {
                    "section@div.container": {
                        "heading@h1.heading": "",
                        "paragraph@p.paragraph": ""
                    }
                }
            },
            "props": {}
        }
    }


def create_example_with_background(example_id: str, bg_type: str = "image") -> Dict[str, Any]:
    """Create example template with background (image/video/overlay)."""
    template = create_example_template(example_id)
    
    if bg_type == "image":
        template["layout"]["props"]["bi"] = "div.background_image"
        template["layout"]["props"]["bo"] = "div.background_overlay"
    elif bg_type == "video":
        template["layout"]["props"]["bv"] = "div.background_video"
        template["layout"]["props"]["bo"] = "div.background_overlay"
    
    return template 