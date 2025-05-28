"""
dc_diagram_fixed.py – works with diagrams 0.24.4
"""

from diagrams import Diagram, Cluster, Edge
from diagrams.onprem.network import Nginx            # grid icon
from diagrams.onprem.monitoring import Datadog       # generic block
from diagrams.generic.compute import Rack            # rack icon
from diagrams.onprem.compute import Server           # placeholder for IT + genset
from diagrams.custom import Custom                   # for generator icon
import os, sys

# --- optional custom icon for the generator -------------------------------
HERE = os.path.dirname(__file__)
GEN_ICON = os.path.join(HERE, "icons", "generator.png")     # supply your own
if os.path.isfile(GEN_ICON):
    Generator = lambda label: Custom(label, GEN_ICON)
else:
    # fallback if no icon file is present
    Generator = lambda label: Server(label + "\n(placeholder)")

# --- draw ------------------------------------------------------------------
with Diagram(
    "Data-centre Power Supply (v0.24.4 node set)",
    show=False,
    direction="LR",
    graph_attr={"splines": "ortho"},
):
    with Cluster("I. Incoming Power"):
        grid = Nginx("Utility Grid")
        microgrid = Custom("Micro-grid\n(Solar)", os.path.join(HERE, "icons", "solar.png"))
        transformer = Datadog("Step-down\nTransformer")
        switchgear = Datadog("Switchgear")

        grid >> Edge(label="HV AC") >> transformer
        microgrid >> Edge(label="Supplement") >> transformer
        transformer >> Edge(label="480/208 V") >> switchgear

    with Cluster("II. Redundancy & Backup"):
        ats = Datadog("ATS")
        ups = Datadog("UPS")
        genset = Generator("Backup\nGenerator")

        switchgear >> ats >> ups
        genset >> Edge(style="dashed") >> ats

    with Cluster("III. Distribution"):
        main_swbd = Datadog("Main Panels")
        floor_pdu = Datadog("Floor PDU")
        rpp = Datadog("RPP")
        busway = Datadog("Busway")

        with Cluster("Server Racks"):
            rack1_pdu = Datadog("Rack PDU 1")
            rack2_pdu = Datadog("Rack PDU 2")
            rack1 = Rack("Rack 1\nServers")
            rack2 = Rack("Rack 2\nServers")

        ups >> main_swbd >> floor_pdu
        floor_pdu >> [rpp, busway]
        rpp >> rack1_pdu >> rack1
        busway >> rack2_pdu >> rack2
