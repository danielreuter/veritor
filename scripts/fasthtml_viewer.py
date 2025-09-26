#!/usr/bin/env python3
"""
FastHTML-based viewer for Veritor workload databases.

Usage:
    python scripts/fasthtml_viewer.py /path/to/database.db

Then open http://localhost:5001 in your browser.
"""

import sys
import json
import webbrowser
from pathlib import Path
from typing import Optional

from fasthtml.common import *

# Try to import Veritor modules
try:
    from veritor.db.api import WorkloadDatabase
    from veritor.db.ir_store import IRRole
except ImportError:
    print("Error: Could not import Veritor modules. Make sure you're running from the project root.")
    sys.exit(1)

# Global database instance
db: Optional[WorkloadDatabase] = None
db_path: str = ""

# FastHTML app setup with modern styling
hdrs = [
    Link(rel="stylesheet", href="https://cdn.jsdelivr.net/npm/@picocss/pico@2/css/pico.min.css"),
    Link(rel="stylesheet", href="https://cdnjs.cloudflare.com/ajax/libs/prism/1.29.0/themes/prism-tomorrow.min.css"),
    Script(src="https://cdnjs.cloudflare.com/ajax/libs/prism/1.29.0/components/prism-core.min.js"),
    Script(src="https://cdnjs.cloudflare.com/ajax/libs/prism/1.29.0/plugins/autoloader/prism-autoloader.min.js"),
    Style("""
        :root { --pico-font-size: 14px; }
        .hero {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            text-align: center;
            padding: 2rem;
            margin-bottom: 2rem;
            border-radius: 8px;
        }
        .stats-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 1rem;
            margin-bottom: 2rem;
        }
        .stat-card {
            text-align: center;
            padding: 1.5rem;
            background: var(--pico-card-background-color);
            border-radius: 8px;
            border: 1px solid var(--pico-card-border-color);
        }
        .stat-number {
            font-size: 2rem;
            font-weight: bold;
            color: var(--pico-primary);
        }
        .graph-card {
            margin-bottom: 1rem;
            border: 1px solid var(--pico-card-border-color);
            border-radius: 8px;
            overflow: hidden;
        }
        .graph-header {
            background: var(--pico-card-sectioning-background-color);
            padding: 1rem;
            border-bottom: 1px solid var(--pico-card-border-color);
        }
        .graph-content {
            padding: 1rem;
        }
        .ir-viewer {
            max-height: 500px;
            overflow-y: auto;
            border: 1px solid var(--pico-card-border-color);
            border-radius: 4px;
        }
        .trace-event {
            padding: 0.5rem;
            margin: 0.25rem 0;
            background: var(--pico-card-sectioning-background-color);
            border-radius: 4px;
            border-left: 4px solid var(--pico-primary);
        }
        .event-time {
            font-family: monospace;
            color: var(--pico-muted-color);
            font-size: 0.9em;
        }
        .nav-tabs {
            display: flex;
            border-bottom: 1px solid var(--pico-card-border-color);
            margin-bottom: 1rem;
        }
        .nav-tab {
            padding: 1rem;
            cursor: pointer;
            border-bottom: 2px solid transparent;
            transition: all 0.2s;
        }
        .nav-tab:hover {
            background: var(--pico-card-sectioning-background-color);
        }
        .nav-tab.active {
            border-bottom-color: var(--pico-primary);
            color: var(--pico-primary);
        }
        .tensor-shape {
            font-family: monospace;
            background: var(--pico-code-background-color);
            padding: 0.2rem 0.4rem;
            border-radius: 3px;
            font-size: 0.9em;
        }
    """)
]

app, rt = fast_app(hdrs=hdrs)

def format_timestamp(timestamp: float) -> str:
    """Format timestamp for display."""
    from datetime import datetime
    try:
        dt = datetime.fromtimestamp(timestamp)
        return dt.strftime("%H:%M:%S.%f")[:-3]  # Include milliseconds
    except:
        return f"{timestamp:.3f}"

def render_overview():
    """Render the overview page."""
    if not db:
        return Div("No database loaded", style="text-align: center; color: var(--pico-muted-color);")

    # Statistics
    stats = [
        ("Graphs", len(db.graphs), "🔄"),
        ("Traces", len(db.traces), "📊"),
        ("Data Bundles", len(db.data_bundles), "💾"),
        ("Challenges", len(db.challenges), "🎯"),
    ]

    stat_cards = [
        Div(
            Div(icon, style="font-size: 2rem; margin-bottom: 0.5rem;"),
            Div(str(count), cls="stat-number"),
            Div(name, style="color: var(--pico-muted-color);"),
            cls="stat-card"
        ) for name, count, icon in stats
    ]

    # Recent activity
    recent_graphs = list(db.graphs.items())[:5]
    graph_list = [
        Div(
            H4(graph_id, style="margin: 0 0 0.5rem 0;"),
            P(f"Type: {graph.metadata.get('model_type', 'unknown')}", style="margin: 0; color: var(--pico-muted-color);"),
            style="margin-bottom: 1rem;"
        ) for graph_id, graph in recent_graphs
    ]

    return Div(
        Div(*stat_cards, cls="stats-grid"),

        H3("Recent Graphs"),
        Div(*graph_list) if graph_list else P("No graphs found", style="color: var(--pico-muted-color);"),

        Details(
            Summary("Database Info"),
            P(f"Database path: {db_path}"),
            P(f"IR Store: {len(getattr(db, 'ir_store', {}).graph_ir_mapping)} graph mappings") if hasattr(db, 'ir_store') else None,
        )
    )

def render_graphs():
    """Render the graphs page."""
    if not db or not db.graphs:
        return P("No graphs found", style="color: var(--pico-muted-color);")

    graph_cards = []
    for graph_id, graph in db.graphs.items():
        # Get IR roles if available
        ir_roles = []
        if hasattr(db, 'ir_store'):
            try:
                ir_roles = db.ir_store.list_ir_for_graph(graph_id)
            except:
                pass

        # Get related traces
        related_traces = [t for t in db.traces.values() if t.graph_id == graph_id]

        # Get related data bundles
        related_data = [d for d in db.data_bundles.values() if d.graph_id == graph_id]

        metadata_items = [
            P(f"{k}: {v}") for k, v in graph.metadata.items()
        ] if graph.metadata else [P("No metadata")]

        graph_card = Div(
            Div(
                H4(graph_id, style="margin: 0;"),
                Small(f"ID: {graph_id}", style="color: var(--pico-muted-color);"),
                cls="graph-header"
            ),
            Div(
                Details(
                    Summary("Metadata"),
                    *metadata_items
                ),
                Details(
                    Summary(f"IR Representations ({len(ir_roles)})"),
                    *[
                        Div(
                            Strong(role),
                            Br(),
                            A(f"View {role} IR", href=f"/ir/{graph_id}/{role}",
                              style="margin-left: 1rem;")
                        ) for role in ir_roles
                    ] if ir_roles else [P("No IR available")]
                ),
                Details(
                    Summary(f"Related Traces ({len(related_traces)})"),
                    *[
                        P(A(trace.id, href=f"/traces#{trace.id}"))
                        for trace in related_traces
                    ] if related_traces else [P("No traces")]
                ),
                Details(
                    Summary(f"Data Bundles ({len(related_data)})"),
                    *[
                        P(A(data.id, href=f"/data#{data.id}"))
                        for data in related_data
                    ] if related_data else [P("No data bundles")]
                ),
                cls="graph-content"
            ),
            cls="graph-card"
        )
        graph_cards.append(graph_card)

    return Div(*graph_cards)

def render_traces():
    """Render the traces page."""
    if not db or not db.traces:
        return P("No traces found", style="color: var(--pico-muted-color);")

    trace_cards = []
    for trace_id, trace in db.traces.items():
        # Event summary
        event_types = {}
        for event in trace.events:
            event_type = event.event_type.value if hasattr(event.event_type, 'value') else str(event.event_type)
            event_types[event_type] = event_types.get(event_type, 0) + 1

        event_summary = ", ".join([f"{count} {etype}" for etype, count in event_types.items()])

        # Recent events (limit to 10)
        recent_events = trace.events[-10:] if len(trace.events) > 10 else trace.events
        event_divs = []

        for event in recent_events:
            event_type = event.event_type.value if hasattr(event.event_type, 'value') else str(event.event_type)
            event_div = Div(
                Div(
                    Strong(event.operation_id or "unknown"),
                    Small(f" ({event_type})", style="margin-left: 0.5rem;"),
                ),
                Div(format_timestamp(event.timestamp), cls="event-time"),
                Div(f"Device: {event.device_id}" if event.device_id else ""),
                cls="trace-event"
            )
            event_divs.append(event_div)

        duration = trace.end_time - trace.start_time if trace.end_time and trace.start_time else 0

        trace_card = Div(
            Div(
                H4(trace_id, style="margin: 0;"),
                P(f"Graph: {trace.graph_id}", style="margin: 0; color: var(--pico-muted-color);"),
                P(f"Duration: {duration:.3f}s | Events: {len(trace.events)}",
                  style="margin: 0; color: var(--pico-muted-color);"),
                cls="graph-header"
            ),
            Div(
                P(f"Event types: {event_summary}"),
                Details(
                    Summary(f"Recent Events ({len(recent_events)} of {len(trace.events)})"),
                    *event_divs
                ),
                cls="graph-content"
            ),
            cls="graph-card",
            id=trace_id
        )
        trace_cards.append(trace_card)

    return Div(*trace_cards)

def render_data():
    """Render the data page."""
    if not db or not db.data_bundles:
        return P("No data bundles found", style="color: var(--pico-muted-color);")

    data_cards = []
    for data_id, data_bundle in db.data_bundles.items():
        # Input shapes
        input_shapes = []
        for name, tensor_data in data_bundle.inputs.items():
            shape_str = f"{name}: {tensor_data.shape}" if hasattr(tensor_data, 'shape') else f"{name}: unknown"
            input_shapes.append(Span(shape_str, cls="tensor-shape"))
            input_shapes.append(Br())

        # Output shapes
        output_shapes = []
        for name, tensor_data in data_bundle.outputs.items():
            shape_str = f"{name}: {tensor_data.shape}" if hasattr(tensor_data, 'shape') else f"{name}: unknown"
            output_shapes.append(Span(shape_str, cls="tensor-shape"))
            output_shapes.append(Br())

        # Activations count
        activation_count = len(data_bundle.activations) if data_bundle.activations else 0

        data_card = Div(
            Div(
                H4(data_id, style="margin: 0;"),
                P(f"Graph: {data_bundle.graph_id}", style="margin: 0; color: var(--pico-muted-color);"),
                cls="graph-header"
            ),
            Div(
                Details(
                    Summary(f"Inputs ({len(data_bundle.inputs)})"),
                    *input_shapes if input_shapes else [P("No inputs")]
                ),
                Details(
                    Summary(f"Outputs ({len(data_bundle.outputs)})"),
                    *output_shapes if output_shapes else [P("No outputs")]
                ),
                P(f"Activations: {activation_count}"),
                P(f"Weights: {len(data_bundle.weights) if data_bundle.weights else 0}"),
                cls="graph-content"
            ),
            cls="graph-card",
            id=data_id
        )
        data_cards.append(data_card)

    return Div(*data_cards)

@rt("/")
def index():
    """Main page with overview."""
    return Html(
        Head(
            Title("🔍 Veritor Database Explorer"),
            *hdrs
        ),
        Body(
            Main(
                Div(
                    Div(
                        H1("🔍 Veritor Database Explorer", style="margin: 0; font-size: 2.5rem;"),
                        P(f"Exploring: {Path(db_path).name}" if db_path else "No database loaded",
                          style="margin: 0; opacity: 0.9;")
                    ),
                    cls="hero"
                ),

                Nav(
                    Div(
                        A("Overview", href="/", cls="nav-tab active"),
                        A("Graphs", href="/graphs", cls="nav-tab"),
                        A("Traces", href="/traces", cls="nav-tab"),
                        A("Data", href="/data", cls="nav-tab"),
                        cls="nav-tabs"
                    )
                ),

                Div(render_overview(), id="content"),

                cls="container"
            ),

            Script("""
                // Simple tab activation
                document.addEventListener('DOMContentLoaded', function() {
                    const currentPath = window.location.pathname;
                    document.querySelectorAll('.nav-tab').forEach(tab => {
                        tab.classList.remove('active');
                        if (tab.getAttribute('href') === currentPath) {
                            tab.classList.add('active');
                        }
                    });
                });
            """)
        )
    )

@rt("/graphs")
def graphs_page():
    """Graphs page."""
    return Html(
        Head(Title("Graphs - Veritor Explorer"), *hdrs),
        Body(
            Main(
                Div(
                    H1("🔄 Computational Graphs"),
                    cls="hero"
                ),

                Nav(
                    Div(
                        A("Overview", href="/", cls="nav-tab"),
                        A("Graphs", href="/graphs", cls="nav-tab active"),
                        A("Traces", href="/traces", cls="nav-tab"),
                        A("Data", href="/data", cls="nav-tab"),
                        cls="nav-tabs"
                    )
                ),

                render_graphs(),

                cls="container"
            )
        )
    )

@rt("/traces")
def traces_page():
    """Traces page."""
    return Html(
        Head(Title("Traces - Veritor Explorer"), *hdrs),
        Body(
            Main(
                Div(
                    H1("📊 Execution Traces"),
                    cls="hero"
                ),

                Nav(
                    Div(
                        A("Overview", href="/", cls="nav-tab"),
                        A("Graphs", href="/graphs", cls="nav-tab"),
                        A("Traces", href="/traces", cls="nav-tab active"),
                        A("Data", href="/data", cls="nav-tab"),
                        cls="nav-tabs"
                    )
                ),

                render_traces(),

                cls="container"
            )
        )
    )

@rt("/data")
def data_page():
    """Data page."""
    return Html(
        Head(Title("Data - Veritor Explorer"), *hdrs),
        Body(
            Main(
                Div(
                    H1("💾 Data Bundles"),
                    cls="hero"
                ),

                Nav(
                    Div(
                        A("Overview", href="/", cls="nav-tab"),
                        A("Graphs", href="/graphs", cls="nav-tab"),
                        A("Traces", href="/traces", cls="nav-tab"),
                        A("Data", href="/data", cls="nav-tab active"),
                        cls="nav-tabs"
                    )
                ),

                render_data(),

                cls="container"
            )
        )
    )

@rt("/ir/{graph_id}/{role}")
def ir_viewer(graph_id: str, role: str):
    """IR viewer page."""
    if not db or not hasattr(db, 'ir_store'):
        return Html(
            Head(Title("IR Viewer - Error"), *hdrs),
            Body(
                Main(
                    H1("Error: No IR store available"),
                    P(A("← Back to Graphs", href="/graphs")),
                    cls="container"
                )
            )
        )

    try:
        ir_text = db.ir_store.get_ir(graph_id, role)
        ir_metadata = db.ir_store.graph_ir_mapping.get(graph_id, {}).get(role, {})
    except Exception as e:
        return Html(
            Head(Title("IR Viewer - Error"), *hdrs),
            Body(
                Main(
                    H1(f"Error loading IR: {e}"),
                    P(A("← Back to Graphs", href="/graphs")),
                    cls="container"
                )
            )
        )

    # Language detection
    language = "mlir"  # Default
    if "stablehlo" in ir_text.lower():
        language = "mlir"
    elif "hlo" in ir_text.lower():
        language = "mlir"

    return Html(
        Head(Title(f"IR: {graph_id} ({role})"), *hdrs),
        Body(
            Main(
                Div(
                    H1(f"📝 IR Viewer: {role}"),
                    P(f"Graph: {graph_id}"),
                    cls="hero"
                ),

                Nav(
                    A("← Back to Graphs", href="/graphs")
                ),

                Details(
                    Summary("Metadata"),
                    *[P(f"{k}: {v}") for k, v in ir_metadata.items()] if ir_metadata else [P("No metadata")]
                ),

                H3("IR Content"),
                Div(
                    Pre(
                        Code(ir_text, cls=f"language-{language}"),
                        style="max-height: 600px; overflow-y: auto;"
                    ),
                    cls="ir-viewer"
                ),

                cls="container"
            ),

            Script("""
                // Trigger Prism highlighting
                if (typeof Prism !== 'undefined') {
                    Prism.highlightAll();
                }
            """)
        )
    )

def main():
    """Main entry point."""
    global db, db_path

    if len(sys.argv) != 2:
        print("Usage: python scripts/fasthtml_viewer.py /path/to/database.db")
        sys.exit(1)

    db_path = sys.argv[1]

    # Load database
    try:
        print(f"Loading database from {db_path}...")
        db = WorkloadDatabase.load(db_path)
        print(f"✅ Loaded database with {len(db.graphs)} graphs, {len(db.traces)} traces")
    except Exception as e:
        print(f"❌ Error loading database: {e}")
        sys.exit(1)

    # Start server
    port = 5001
    url = f"http://localhost:{port}"
    print(f"🚀 Starting Veritor Explorer at {url}")
    print(f"📊 Database: {Path(db_path).name}")
    print(f"📁 Path: {db_path}")
    print("\nPress Ctrl+C to stop the server")

    # Open browser
    try:
        webbrowser.open(url)
    except:
        pass

    # Run server
    serve(port=port)

if __name__ == "__main__":
    main()