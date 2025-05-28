# dc_diagram.py
import sys
print(f"--- Diagnostic Information ---")
print(f"Python Executable: {sys.executable}")
print(f"Python Version: {sys.version}")
print(f"Python Path (sys.path):")
for p in sys.path:
    print(f"  - {p}")
print(f"----------------------------")

try:
    print("\nAttempting to import 'diagrams' components...")
    from diagrams import Diagram, Cluster, Edge
    print("  Successfully imported: Diagram, Cluster, Edge")

    from diagrams.onprem.network import Nginx
    print("  Successfully imported: onprem.network.Nginx")

    # Key problematic import:
    from diagrams.onprem.power import UPS, PDU, Rack, Generator
    print("  Successfully imported: onprem.power components (UPS, PDU, Rack, Generator)")

    from diagrams.onprem.compute import Server
    print("  Successfully imported: onprem.compute.Server")

    from diagrams.onprem.monitoring import Datadog
    print("  Successfully imported: onprem.monitoring.Datadog")
    print("All core 'diagrams' imports seem successful!\n")

    # Placeholder for SolarPanel to avoid GCP dependency until core issue is fixed
    # If you later install diagrams-gcp, you can use:
    # from diagrams.gcp.energy import SolarPanel
    class SolarPanel: # Simple placeholder class
        _default_graph_attrs = {} # Mimic DiagramNode attribute
        _default_node_attrs = {}  # Mimic DiagramNode attribute
        def __init__(self, label, **attrs):
            self.label = label
            self.attrs = attrs
            print(f"    [Placeholder] SolarPanel '{label}' created")
        def __rshift__(self, other): # Allow use with >> operator (for edge creation)
            return self # Simplistic, real library does more
        def __lshift__(self, other): # Allow use with << operator
            return self # Simplistic
        def __repr__(self):
            return f"SolarPanel(label='{self.label}')"
        # Add a dummy method that might be called by Diagram context
        def _get_graph_attrs(self):
            return {}


except ModuleNotFoundError as e:
    print(f"\n❌ ERROR: A ModuleNotFoundError occurred: {e}")
    print("This means Python could not find a specified module.")
    print("This often happens if the package isn't installed correctly in the environment Python is using.")
    print("Please ensure 'diagrams' is correctly installed in your virtual environment by following these steps in your venv:")
    print("  1. Ensure your venv is activated.")
    print("  2. python -m pip install --upgrade pip")
    print("  3. pip uninstall diagrams -y")
    print("  4. pip install diagrams==0.24.4 --no-cache-dir")
    print("If the error persists, the 'Python Path' printed above might not include the correct 'site-packages' directory for your venv.")
    sys.exit(1)
except ImportError as e:
    print(f"\n❌ ERROR: An ImportError occurred: {e}")
    print("This means a module was found, but something it tried to import failed.")
    sys.exit(1)


print("Proceeding to create diagram object...\n")
with Diagram("Data Center Power Supply Schematic", show=False, direction="LR", graph_attr={"splines":"ortho"}):

    with Cluster("I. Incoming Power & Transformation"):
        grid = Nginx("Utility Grid")
        microgrid = SolarPanel("Microgrid (Solar)") # Using placeholder
        transformer = Datadog("Transformers\n(Step-Down)")
        switchgear = Datadog("Switchgear")

        grid >> Edge(label="High-Voltage AC") >> transformer
        microgrid >> Edge(label="Supplementary Power") >> transformer
        transformer >> Edge(label="Lower Voltage AC\n(e.g., 480V)") >> switchgear

    with Cluster("II. Redundancy and Backup Systems"):
        ats = Datadog("Automatic Transfer\nSwitch (ATS)")
        with Cluster("UPS System"):
            ups_system = UPS("UPS Unit\n(with Batteries & Inverter)")

        generator = Generator("Backup Generators")

        switchgear >> Edge(label="Primary Power") >> ats
        ats >> Edge(label="To UPS / Load") >> ups_system
        generator >> Edge(label="Backup Power") >> ats

    with Cluster("III. Power Distribution within Data Center"):
        main_switchboard = Datadog("Main Switchboards/\nElectrical Panels")
        floor_pdu = PDU("Floor PDUs")

        with Cluster("Local Distribution"):
            rpp = Datadog("Remote Power\nPanels (RPPs)")
            busway = Datadog("Busways")

        with Cluster("Server Racks"):
            rack_pdu1 = PDU("Rack PDU 1")
            rack_pdu2 = PDU("Rack PDU 2")
            it_equipment1 = Server("IT Equipment 1")
            it_equipment2 = Server("IT Equipment (Rack 2)")

        ups_system >> Edge(label="Conditioned/\nBattery Power") >> main_switchboard
        main_switchboard >> floor_pdu
        floor_pdu >> rpp
        floor_pdu >> busway
        rpp >> rack_pdu1
        busway >> rack_pdu2
        rack_pdu1 >> it_equipment1
        rack_pdu2 >> it_equipment2

print("\n✅ Diagram object created successfully.")
print("If Graphviz is installed and configured, a PNG file should be generated.")