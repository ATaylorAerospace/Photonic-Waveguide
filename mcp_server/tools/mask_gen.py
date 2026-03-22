"""GDSII mask generation using gdsfactory."""
import os
import gdsfactory as gf
from mcp_server.config import GDS_OUTPUT_DIR
from mcp_server.schemas.waveguide import MaskGenInput, MaskGenOutput


class MaskGenerator:
    """Wraps gdsfactory to produce GDSII layout files."""

    def __init__(self, output_dir: str = GDS_OUTPUT_DIR):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def generate(self, params: MaskGenInput) -> MaskGenOutput:
        """Generate a foundry-ready GDSII file from waveguide parameters."""
        c = gf.Component("photonic_waveguide_design")
        xs = gf.cross_section.cross_section(width=params.width_um, layer=params.layer)
        length_um = params.length_mm * 1000.0

        straight = gf.components.straight(length=length_um, cross_section=xs)
        straight_ref = c.add_ref(straight)

        taper_in = gf.components.taper(
            length=params.taper_length_um,
            width1=0.2,
            width2=params.width_um,
            layer=params.layer,
        )
        taper_in_ref = c.add_ref(taper_in)
        taper_in_ref.connect("o2", straight_ref.ports["o1"])

        taper_out = gf.components.taper(
            length=params.taper_length_um,
            width1=params.width_um,
            width2=0.2,
            layer=params.layer,
        )
        taper_out_ref = c.add_ref(taper_out)
        taper_out_ref.connect("o1", straight_ref.ports["o2"])

        if params.io_type == "grating_coupler":
            gc_in = gf.components.grating_coupler_elliptical_trenches(
                taper_length=15.0,
                wavelength=1.55,
                cross_section=xs,
            )
            gc_in_ref = c.add_ref(gc_in)
            gc_in_ref.connect("o1", taper_in_ref.ports["o1"])

            gc_out = gf.components.grating_coupler_elliptical_trenches(
                taper_length=15.0,
                wavelength=1.55,
                cross_section=xs,
            )
            gc_out_ref = c.add_ref(gc_out)
            gc_out_ref.connect("o1", taper_out_ref.ports["o2"])

        output_path = os.path.join(self.output_dir, params.output_filename)
        c.write_gds(output_path)

        bbox = c.bbox
        bounding_box = ((float(bbox[0][0]), float(bbox[0][1])),
                        (float(bbox[1][0]), float(bbox[1][1])))

        return MaskGenOutput(
            gds_file_path=output_path,
            cell_name=c.name,
            total_length_um=length_um + 2 * params.taper_length_um,
            num_bends=0,
            bounding_box=bounding_box,
        )
