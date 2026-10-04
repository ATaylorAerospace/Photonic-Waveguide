"""GDSII mask generation using gdsfactory."""
import hashlib
import json
import os
import threading

import gdsfactory as gf

from mcp_server.config import GDS_OUTPUT_DIR
from mcp_server.schemas.waveguide import MaskGenInput, MaskGenOutput

# gdsfactory builds every component into one process-global layout, which is
# not safe to mutate from the MCP server's concurrent worker threads.
_LAYOUT_LOCK = threading.Lock()


def _ensure_pdk() -> None:
    """gdsfactory >= 9 no longer activates a PDK implicitly."""
    try:
        gf.get_active_pdk()
    except ValueError:
        gf.gpdk.get_generic_pdk().activate()


class MaskGenerator:
    """Wraps gdsfactory to produce GDSII layout files."""

    def __init__(self, output_dir: str = GDS_OUTPUT_DIR):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        _ensure_pdk()

    @staticmethod
    def _design_digest(params: MaskGenInput) -> str:
        return hashlib.sha256(
            json.dumps(params.model_dump(), sort_keys=True).encode()
        ).hexdigest()[:12]

    @classmethod
    def _output_filename(cls, params: MaskGenInput) -> str:
        """Derive a geometry-unique filename from the requested one.

        Different designs sharing the default filename would otherwise
        overwrite each other's GDS file while the DynamoDB cache keeps
        serving the old path — returning a file whose contents belong to
        another design.
        """
        stem, ext = os.path.splitext(os.path.basename(params.output_filename))
        return f"{stem}_{cls._design_digest(params)}{ext or '.gds'}"

    def generate(self, params: MaskGenInput) -> MaskGenOutput:
        """Generate a foundry-ready GDSII file from waveguide parameters."""
        with _LAYOUT_LOCK:
            return self._generate(params)

    def _generate(self, params: MaskGenInput) -> MaskGenOutput:
        # The layout refuses to create a second cell with an existing name, so
        # the top cell must be unique per design and removed once written.
        c = gf.Component(f"photonic_waveguide_{self._design_digest(params)}")
        try:
            xs = gf.cross_section.cross_section(width=params.width_um, layer=params.layer)
            length_um = params.length_mm * 1000.0

            straight = gf.components.straight(length=length_um, cross_section=xs)
            straight_ref = c.add_ref(straight)

            if params.io_type == "grating_coupler":
                # The coupler's own taper brings the mode to full width, so it
                # attaches straight to the waveguide ends.
                for port_name in ("o1", "o2"):
                    gc = gf.components.grating_coupler_elliptical_trenches(
                        taper_length=15.0,
                        wavelength=1.55,
                        cross_section=xs,
                    )
                    gc_ref = c.add_ref(gc)
                    gc_ref.connect("o1", straight_ref.ports[port_name])
            else:
                # Edge coupler: inverse tapers down to a 0.2 um tip at each facet.
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

            output_path = os.path.join(self.output_dir, self._output_filename(params))
            c.write_gds(output_path)

            # gdsfactory >= 8 (kfactory-based): bbox is a method returning a DBox.
            bbox = c.dbbox()
            bounding_box = ((float(bbox.left), float(bbox.bottom)),
                            (float(bbox.right), float(bbox.top)))
            cell_name = c.name
        finally:
            # Drop the top cell so a long-running server does not accumulate
            # one layout cell per request; sub-components stay cached for reuse.
            c.delete()

        return MaskGenOutput(
            gds_file_path=output_path,
            cell_name=cell_name,
            total_length_um=float(bbox.right - bbox.left),
            bounding_box=bounding_box,
        )
