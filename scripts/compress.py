from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any
import logging
import numpy as np
import struct
from plyfile import PlyData
import math

@dataclass
class SplatConfig:
    """Configuration constants for splat conversion"""
    SH_C0: float = 0.28209479177387814
    BYTES_PER_SPLAT: int = 32
    REQUIRED_PROPERTIES: List[str] = (
        'x', 'y', 'z',
        'scale_0', 'scale_1', 'scale_2',
        'rot_0', 'rot_1', 'rot_2', 'rot_3',
        'f_dc_0', 'f_dc_1', 'f_dc_2',
        'opacity'
    )

@dataclass
class Splat:
    """Represents a single splat's data"""
    position: tuple[float, float, float]
    scale: tuple[float, float, float]
    color: tuple[int, int, int, int]
    rotation: tuple[int, int, int, int]

    def to_bytes(self) -> bytearray:
        """Convert splat data to binary format"""
        buffer = bytearray(SplatConfig.BYTES_PER_SPLAT)
        
        # Position (xyz) as float32
        for i, pos in enumerate(self.position):
            struct.pack_into('<f', buffer, i * 4, pos)
        
        # Scale (xyz) as float32
        for i, scale in enumerate(self.scale):
            struct.pack_into('<f', buffer, 12 + i * 4, math.exp(scale))
        
        # Color (rgba) as uint8
        struct.pack_into('<BBBB', buffer, 24, *self.color)
        
        # Rotation (quaternion) as uint8
        struct.pack_into('<BBBB', buffer, 28, *self.rotation)
        
        return buffer

class PLYConverter:
    def __init__(self, config: SplatConfig = SplatConfig()):
        self.config = config
        self.logger = logging.getLogger(__name__)

    @staticmethod
    def load_ply(filepath: Path) -> PlyData:
        """Load PLY file and return vertex data"""
        logging.info(f"Loading PLY file: {filepath}")
        try:
            plydata = PlyData.read(filepath)
            vertex = plydata['vertex']
            logging.info(f"Successfully loaded PLY file with {len(vertex)} vertices")
            return vertex
        except Exception as e:
            logging.error(f"Failed to load PLY file: {e}")
            raise

    def verify_properties(self, available_props: List[str]) -> None:
        """Verify all required properties exist in PLY file"""
        missing_props = set(self.config.REQUIRED_PROPERTIES) - set(available_props)
        if missing_props:
            raise ValueError(f"Missing required properties: {missing_props}")

    @staticmethod
    def clamp(x: float) -> int:
        """Clamp value between 0 and 255"""
        return max(0, min(255, int(x)))

    def process_vertex_row(self, row: Any, prop_indices: Dict[str, int]) -> Splat:
        """Convert a vertex row to Splat object"""
        try:
            # Position
            position = tuple(float(row[prop_indices[f]]) for f in ['x', 'y', 'z'])
            
            # Scale
            scale = tuple(float(row[prop_indices[f]]) for f in ['scale_0', 'scale_1', 'scale_2'])
            
            # Color with spherical harmonics
            color = (
                self.clamp((0.5 + self.config.SH_C0 * float(row[prop_indices['f_dc_0']])) * 255),
                self.clamp((0.5 + self.config.SH_C0 * float(row[prop_indices['f_dc_1']])) * 255),
                self.clamp((0.5 + self.config.SH_C0 * float(row[prop_indices['f_dc_2']])) * 255),
                self.clamp((1 / (1 + math.exp(-float(row[prop_indices['opacity']])))) * 255)
            )
            
            # Rotation
            rotation = tuple(
                self.clamp(float(row[prop_indices[f'rot_{i}']]) * 128 + 128)
                for i in range(4)
            )

            return Splat(position, scale, color, rotation)
            
        except Exception as e:
            logging.error(f"Error processing vertex row: {row}")
            raise ValueError(f"Failed to process vertex row: {e}")

    def serialize_splat(self, input_path: Path, output_path: Path) -> None:
        """Convert PLY file to binary splat format"""
        vertex_data = self.load_ply(input_path)
        total_splats = len(vertex_data)
        
        # Get property indices
        prop_indices = {p.name: i for i, p in enumerate(vertex_data.properties)}
        self.verify_properties([p.name for p in vertex_data.properties])
        
        # Process splats
        buffer = bytearray(total_splats * self.config.BYTES_PER_SPLAT)
        for i in range(total_splats):
            if i > 0 and i % 10000 == 0:
                logging.info(f"Processed {i}/{total_splats} splats ({(i/total_splats)*100:.1f}%)")
            
            splat = self.process_vertex_row(vertex_data[i], prop_indices)
            buffer[i * self.config.BYTES_PER_SPLAT:(i + 1) * self.config.BYTES_PER_SPLAT] = splat.to_bytes()

        # Write to file
        logging.info(f"Writing {len(buffer)} bytes to {output_path}")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(buffer)

def main():
    import argparse
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    parser = argparse.ArgumentParser(description='Convert PLY file to binary splat format')
    parser.add_argument('input', type=Path, help='Input PLY file path')
    parser.add_argument('output', type=Path, help='Output splat file path')
    
    args = parser.parse_args()
    
    try:
        converter = PLYConverter()
        converter.serialize_splat(args.input, args.output)
        logging.info("Conversion completed successfully")
    except Exception as e:
        logging.error(f"Conversion failed: {e}")
        exit(1)

if __name__ == "__main__":
    main()