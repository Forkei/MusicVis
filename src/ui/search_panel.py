"""Search panel UI with imgui."""

from imgui_bundle import imgui


class SearchPanel:
    """YouTube search bar and results list."""

    def __init__(self):
        self.query = ""
        self.results: list[dict] = []
        self.selected_index: int = -1
        self.searching = False
        self.error_msg = ""

    def draw(self) -> dict | None:
        """Draw the search panel. Returns selected result dict or None."""
        selected = None

        imgui.set_next_window_pos((20, 20), imgui.Cond_.once)
        imgui.set_next_window_size((400, 500), imgui.Cond_.once)

        expanded, _ = imgui.begin("Search YouTube", None, imgui.WindowFlags_.no_collapse)
        if expanded:
            # Search input
            imgui.set_next_item_width(-80)
            changed, self.query = imgui.input_text(
                "##search", self.query, imgui.InputTextFlags_.enter_returns_true
            )
            trigger_search = changed

            imgui.same_line()
            if imgui.button("Search") or trigger_search:
                if self.query.strip():
                    self.searching = True
                    self.error_msg = ""

            if self.searching:
                imgui.text("Searching...")

            if self.error_msg:
                imgui.text_colored((1.0, 0.3, 0.3, 1.0), self.error_msg)

            imgui.separator()

            # Results list
            if self.results:
                for i, r in enumerate(self.results):
                    title = r.get("title", "Unknown")
                    channel = r.get("channel", "")
                    dur = r.get("duration_str", "?:??")

                    label = f"{title}\n  {channel} - {dur}"
                    is_selected = i == self.selected_index

                    if imgui.selectable(f"{title}##{i}", is_selected)[0]:
                        self.selected_index = i
                        selected = r

                    # Show details on same line area
                    imgui.same_line(imgui.get_window_width() - 60)
                    imgui.text_disabled(dur)

                    if channel:
                        imgui.text_disabled(f"  {channel}")
            elif not self.searching:
                imgui.text_disabled("Search for a song to get started")

        imgui.end()
        return selected
