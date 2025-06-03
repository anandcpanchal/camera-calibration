import tkinter as tk
from tkinter import ttk, messagebox
import cv2
import numpy as np
from PIL import Image, ImageTk, ImageDraw
import ast  # For safely evaluating string literals

import matplotlib

matplotlib.use('TkAgg')
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


class HomographyVisualizerApp:
    def __init__(self, master):
        self.master = master
        master.title("Interactive Homography Visualizer")
        master.geometry("1000x950")

        self.canvas_width = 280
        self.canvas_height = 280
        self.grid_step = 20
        self.point_radius = 4
        self.line_width = 2

        # --- Point count ---
        self.num_points_var = tk.IntVar(value=4)
        self.max_points = 16

        # --- Input Mode ---
        self.input_mode_var = tk.StringVar(value="individual")  # "individual" or "raw"

        self.src_points_entries = []
        self.dst_points_entries = []

        self.src_point_entry_frames_container = None  # For individual fields
        self.dst_point_entry_frames_container = None  # For individual fields

        self.src_raw_text = None  # For raw text input
        self.dst_raw_text = None  # For raw text input
        self.src_raw_text_frame = None  # Frame to hold raw text widget for source
        self.dst_raw_text_frame = None  # Frame to hold raw text widget for destination

        self.homography_matrix_entries = []
        self.H = np.eye(3)

        self.base_grid_image_pil = self._create_grid_pil(self.canvas_width, self.canvas_height, self.grid_step)
        self.base_grid_image_cv = self._pil_to_cv2(self.base_grid_image_pil.copy())
        self.show_vectors_var = tk.BooleanVar(value=False)

        self.default_src_pts_data = [
            (50, 50), (200, 50), (200, 200), (50, 200),
            (25, 125), (225, 125), (125, 25), (125, 225),
            (75, 75), (175, 175)
        ]
        self.default_dst_pts_data = [
            (70, 70), (230, 60), (240, 180), (60, 190),
            (45, 135), (215, 145), (135, 45), (145, 215),
            (85, 85), (185, 185)
        ]

        self.left_canvas = None
        self.left_scrollbar = None
        self.scrollable_left_frame = None

        self._setup_ui()
        self._update_homography_display()  # Initialize H matrix entries
        # self._draw_initial_canvases() # Removed, handled by _trigger_redraw_visuals flow

        self._on_compute_homography_from_points()  # Computes H from defaults and triggers full redraw

    def _pil_to_cv2(self, pil_image):
        open_cv_image = np.array(pil_image)
        if len(open_cv_image.shape) == 3 and open_cv_image.shape[2] == 3:
            open_cv_image = open_cv_image[:, :, ::-1].copy()
        elif len(open_cv_image.shape) == 3 and open_cv_image.shape[2] == 4:
            open_cv_image = cv2.cvtColor(open_cv_image, cv2.COLOR_RGBA2BGRA)
        return open_cv_image

    def _cv2_to_pil(self, cv_image):
        if len(cv_image.shape) == 3 and cv_image.shape[2] == 3:
            cv_image_rgb = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        elif len(cv_image.shape) == 3 and cv_image.shape[2] == 4:
            cv_image_rgb = cv2.cvtColor(cv_image, cv2.COLOR_BGRA2RGBA)
        else:
            cv_image_rgb = cv2.cvtColor(cv_image, cv2.COLOR_GRAY2RGB)
        return Image.fromarray(cv_image_rgb)

    def _create_grid_pil(self, width, height, step):
        image = Image.new("RGB", (width, height), "white")
        draw = ImageDraw.Draw(image)
        for x in range(0, width, step):
            draw.line([(x, 0), (x, height)], fill="lightgray")
        for y in range(0, height, step):
            draw.line([(0, y), (width, y)], fill="lightgray")
        return image

    def _setup_ui(self):
        main_frame = ttk.Frame(self.master, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)

        left_panel_container = ttk.Frame(main_frame, padding="10")
        left_panel_container.pack(side=tk.LEFT, fill=tk.Y, padx=5)

        self.left_canvas = tk.Canvas(left_panel_container, borderwidth=0)
        self.left_scrollbar = ttk.Scrollbar(left_panel_container, orient="vertical", command=self.left_canvas.yview)
        self.scrollable_left_frame = ttk.Frame(self.left_canvas)

        self.scrollable_left_frame.bind(
            "<Configure>",
            lambda e: self.left_canvas.configure(
                scrollregion=self.left_canvas.bbox("all")
            )
        )

        self.left_canvas.create_window((0, 0), window=self.scrollable_left_frame, anchor="nw")
        self.left_canvas.configure(yscrollcommand=self.left_scrollbar.set)

        self.left_canvas.pack(side="left", fill="both", expand=True)
        self.left_scrollbar.pack(side="right", fill="y")

        # --- Point Count Selector ---
        point_count_frame = ttk.Frame(self.scrollable_left_frame)
        point_count_frame.pack(pady=5, fill=tk.X)
        ttk.Label(point_count_frame, text="Number of Points:").pack(side=tk.LEFT, padx=(0, 5))
        self.point_count_spinbox = ttk.Spinbox(
            point_count_frame, from_=4, to=self.max_points, textvariable=self.num_points_var,
            width=3, command=self._on_num_points_changed, state='readonly'
        )
        self.point_count_spinbox.pack(side=tk.LEFT)
        self.num_points_var.trace_add("write", self._on_num_points_changed_by_var)

        # --- Input Mode Selector ---
        input_mode_frame = ttk.LabelFrame(self.scrollable_left_frame, text="Point Input Mode", padding="5")
        input_mode_frame.pack(pady=5, fill=tk.X)
        ttk.Radiobutton(input_mode_frame, text="Individual Fields", variable=self.input_mode_var,
                        value="individual", command=self._toggle_input_mode).pack(side=tk.LEFT, padx=5)
        ttk.Radiobutton(input_mode_frame, text="Raw Text", variable=self.input_mode_var,
                        value="raw", command=self._toggle_input_mode).pack(side=tk.LEFT, padx=5)

        # --- Source Points Area ---
        self.src_points_main_frame = ttk.LabelFrame(self.scrollable_left_frame, text="Source Points (x, y)",
                                                    padding="10")
        self.src_points_main_frame.pack(pady=5, fill=tk.X)
        self.src_point_entry_frames_container = ttk.Frame(self.src_points_main_frame)
        self.src_point_entry_frames_container.pack(fill=tk.X)
        self.src_raw_text_frame = ttk.Frame(self.src_points_main_frame)
        self.src_raw_text = tk.Text(self.src_raw_text_frame, height=5, width=25, wrap=tk.WORD)
        self.src_raw_text.pack(fill=tk.BOTH, expand=True)

        # --- Destination Points Area ---
        self.dst_points_main_frame = ttk.LabelFrame(self.scrollable_left_frame, text="Destination Points (x', y')",
                                                    padding="10")
        self.dst_points_main_frame.pack(pady=5, fill=tk.X)
        self.dst_point_entry_frames_container = ttk.Frame(self.dst_points_main_frame)
        self.dst_point_entry_frames_container.pack(fill=tk.X)
        self.dst_raw_text_frame = ttk.Frame(self.dst_points_main_frame)
        self.dst_raw_text = tk.Text(self.dst_raw_text_frame, height=5, width=25, wrap=tk.WORD)
        self.dst_raw_text.pack(fill=tk.BOTH, expand=True)

        self._create_point_entry_rows(self.num_points_var.get())

        vector_checkbox = ttk.Checkbutton(self.scrollable_left_frame, text="Show Vectors (2D Origin)",
                                          variable=self.show_vectors_var, command=self._trigger_redraw_visuals)
        vector_checkbox.pack(pady=5)

        compute_button = ttk.Button(self.scrollable_left_frame, text="Compute H from Points",
                                    command=self._on_compute_homography_from_points)
        compute_button.pack(pady=(10, 0))

        matrix_frame = ttk.LabelFrame(self.scrollable_left_frame, text="Homography Matrix (H) - Editable", padding="10")
        matrix_frame.pack(pady=5, fill=tk.X)
        for r in range(3):
            row_frame = ttk.Frame(matrix_frame)
            row_frame.pack()
            row_entries = []
            for c in range(3):
                entry_h = ttk.Entry(row_frame, width=8, justify='right')
                entry_h.insert(0, f"{self.H[r, c]:.4f}")
                entry_h.pack(side=tk.LEFT, padx=2, pady=2)
                row_entries.append(entry_h)
            self.homography_matrix_entries.append(row_entries)

        apply_h_button = ttk.Button(self.scrollable_left_frame, text="Apply Manual H & Update Dst Pts",
                                    command=self._on_apply_manual_homography)
        apply_h_button.pack(pady=(0, 10))

        # --- Right Panel (Visualizations) ---
        right_panel = ttk.Frame(main_frame, padding="10")
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5)
        top_right_panel = ttk.Frame(right_panel);
        top_right_panel.pack(fill=tk.X, expand=False)
        src_canvas_frame = ttk.LabelFrame(top_right_panel, text="Source Plane", padding="5")
        src_canvas_frame.pack(pady=5, padx=5, side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.source_canvas = tk.Canvas(src_canvas_frame, width=self.canvas_width, height=self.canvas_height, bg="white",
                                       relief="ridge", borderwidth=2)
        self.source_canvas.pack()
        dst_canvas_frame = ttk.LabelFrame(top_right_panel, text="Destination Plane (Warped)", padding="5")
        dst_canvas_frame.pack(pady=5, padx=5, side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.destination_canvas = tk.Canvas(dst_canvas_frame, width=self.canvas_width, height=self.canvas_height,
                                            bg="white", relief="ridge", borderwidth=2)
        self.destination_canvas.pack()
        bottom_right_panel = ttk.LabelFrame(right_panel, text="3D Visualization", padding="5")
        bottom_right_panel.pack(fill=tk.BOTH, expand=True, pady=10)
        self.fig_3d = Figure(figsize=(6, 5), dpi=100)
        self.ax_3d = self.fig_3d.add_subplot(111, projection='3d')
        self.canvas_3d_widget = FigureCanvasTkAgg(self.fig_3d, master=bottom_right_panel)
        self.canvas_3d_widget.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # Moved to the end of _setup_ui to ensure all widgets (like canvases) exist
        self._toggle_input_mode()

    def _toggle_input_mode(self):
        mode = self.input_mode_var.get()
        num_pts = self.num_points_var.get()

        if mode == "individual":
            self.src_point_entry_frames_container.pack(fill=tk.X)
            self.dst_point_entry_frames_container.pack(fill=tk.X)
            self.src_raw_text_frame.pack_forget()
            self.dst_raw_text_frame.pack_forget()

            src_pts_from_raw = self._parse_raw_text_points(self.src_raw_text.get("1.0", tk.END).strip(), num_pts)
            if src_pts_from_raw is not None:
                self._update_individual_point_entries(self.src_points_entries, src_pts_from_raw)

            dst_pts_from_raw = self._parse_raw_text_points(self.dst_raw_text.get("1.0", tk.END).strip(), num_pts)
            if dst_pts_from_raw is not None:
                self._update_individual_point_entries(self.dst_points_entries, dst_pts_from_raw)

        elif mode == "raw":
            self.src_point_entry_frames_container.pack_forget()
            self.dst_point_entry_frames_container.pack_forget()
            self.src_raw_text_frame.pack(fill=tk.BOTH, expand=True)
            self.dst_raw_text_frame.pack(fill=tk.BOTH, expand=True)

            current_src_pts_ind = self._get_individual_points(self.src_points_entries)
            if current_src_pts_ind is not None:
                self.src_raw_text.delete("1.0", tk.END)
                self.src_raw_text.insert("1.0", self._format_points_for_raw_text(current_src_pts_ind))

            current_dst_pts_ind = self._get_individual_points(self.dst_points_entries)
            if current_dst_pts_ind is not None:
                self.dst_raw_text.delete("1.0", tk.END)
                self.dst_raw_text.insert("1.0", self._format_points_for_raw_text(current_dst_pts_ind))

        self._trigger_redraw_visuals()

    def _format_points_for_raw_text(self, points_array):
        if points_array is None:
            return ""
        return repr(points_array.tolist())

    def _parse_raw_text_points(self, raw_text_str, expected_num_points):
        if not raw_text_str.strip():
            return None
        try:
            parsed_list = ast.literal_eval(raw_text_str)
            if not isinstance(parsed_list, list):
                return None

            if len(parsed_list) != expected_num_points:
                return None

            points = []
            for item in parsed_list:
                if isinstance(item, (list, tuple)) and len(item) == 2:
                    try:
                        points.append((float(item[0]), float(item[1])))
                    except (ValueError, TypeError):
                        return None
                else:
                    return None
            return np.array(points, dtype=np.float32)
        except (SyntaxError, ValueError, TypeError) as e:
            return None

    def _create_point_entry_rows(self, num_points_to_create):
        for widget in self.src_point_entry_frames_container.winfo_children(): widget.destroy()
        self.src_points_entries.clear()
        for widget in self.dst_point_entry_frames_container.winfo_children(): widget.destroy()
        self.dst_points_entries.clear()

        for i in range(num_points_to_create):
            row_frame = ttk.Frame(self.src_point_entry_frames_container);
            row_frame.pack()
            ttk.Label(row_frame, text=f"P{i + 1}:").pack(side=tk.LEFT)
            ex = ttk.Entry(row_frame, width=5);
            ex.insert(0, str(self.default_src_pts_data[i % len(self.default_src_pts_data)][0]));
            ex.pack(side=tk.LEFT, padx=2)
            ey = ttk.Entry(row_frame, width=5);
            ey.insert(0, str(self.default_src_pts_data[i % len(self.default_src_pts_data)][1]));
            ey.pack(side=tk.LEFT, padx=2)
            self.src_points_entries.append((ex, ey))

        for i in range(num_points_to_create):
            row_frame = ttk.Frame(self.dst_point_entry_frames_container);
            row_frame.pack()
            ttk.Label(row_frame, text=f"P'{i + 1}:").pack(side=tk.LEFT)
            ex = ttk.Entry(row_frame, width=5);
            ex.insert(0, str(self.default_dst_pts_data[i % len(self.default_dst_pts_data)][0]));
            ex.pack(side=tk.LEFT, padx=2)
            ey = ttk.Entry(row_frame, width=5);
            ey.insert(0, str(self.default_dst_pts_data[i % len(self.default_dst_pts_data)][1]));
            ey.pack(side=tk.LEFT, padx=2)
            self.dst_points_entries.append((ex, ey))

    def _on_num_points_changed_by_var(self, *args):
        try:
            new_num_points_from_var = self.num_points_var.get()
            if new_num_points_from_var < 4: self.num_points_var.set(4); return
            if new_num_points_from_var > self.max_points: self.num_points_var.set(self.max_points); return

            self._create_point_entry_rows(new_num_points_from_var)
            if self.input_mode_var.get() == "raw":
                self._transfer_individual_to_raw()
            self._trigger_redraw_visuals()
        except tk.TclError:
            pass

    def _on_num_points_changed(self):
        try:
            new_num_points = self.num_points_var.get()
            self._create_point_entry_rows(new_num_points)
            if self.input_mode_var.get() == "raw":
                self._transfer_individual_to_raw()
            self._trigger_redraw_visuals()
        except tk.TclError:
            pass

    def _get_individual_points(self, point_entries_list):
        pts = []
        num_expected_points = len(point_entries_list)
        if num_expected_points == 0: return None

        try:
            for i in range(num_expected_points):
                ex, ey = point_entries_list[i]
                pts.append((float(ex.get()), float(ey.get())))
        except ValueError:
            return None
        return np.array(pts, dtype=np.float32) if pts else None

    def _get_current_src_points(self):
        mode = self.input_mode_var.get()
        num_pts_expected = self.num_points_var.get()
        if mode == "individual":
            return self._get_individual_points(self.src_points_entries)
        elif mode == "raw":
            raw_data = self.src_raw_text.get("1.0", tk.END).strip()
            parsed = self._parse_raw_text_points(raw_data, num_pts_expected)
            if parsed is None and raw_data:
                messagebox.showwarning("Raw Input Parse Error",
                                       "Source points raw text is invalid or does not match expected point count.")
            return parsed
        return None

    def _get_current_dst_points(self):
        mode = self.input_mode_var.get()
        num_pts_expected = self.num_points_var.get()
        if mode == "individual":
            return self._get_individual_points(self.dst_points_entries)
        elif mode == "raw":
            raw_data = self.dst_raw_text.get("1.0", tk.END).strip()
            parsed = self._parse_raw_text_points(raw_data, num_pts_expected)
            if parsed is None and raw_data:
                messagebox.showwarning("Raw Input Parse Error",
                                       "Destination points raw text is invalid or does not match expected point count.")
            return parsed
        return None

    def _update_individual_point_entries(self, point_entries_ui_list, points_data_array):
        if points_data_array is None: return
        num_to_update = min(len(point_entries_ui_list), len(points_data_array))
        for i in range(num_to_update):
            ex, ey = point_entries_ui_list[i]
            ex.delete(0, tk.END);
            ex.insert(0, f"{points_data_array[i, 0]:.2f}")
            ey.delete(0, tk.END);
            ey.insert(0, f"{points_data_array[i, 1]:.2f}")

    def _update_src_points_display(self, points_data_array):
        mode = self.input_mode_var.get()
        if mode == "individual":
            self._update_individual_point_entries(self.src_points_entries, points_data_array)
        elif mode == "raw":
            self.src_raw_text.delete("1.0", tk.END)
            if points_data_array is not None:
                self.src_raw_text.insert("1.0", self._format_points_for_raw_text(points_data_array))

    def _update_dst_points_display(self, points_data_array):
        mode = self.input_mode_var.get()
        if mode == "individual":
            self._update_individual_point_entries(self.dst_points_entries, points_data_array)
        elif mode == "raw":
            self.dst_raw_text.delete("1.0", tk.END)
            if points_data_array is not None:
                self.dst_raw_text.insert("1.0", self._format_points_for_raw_text(points_data_array))

    def _transfer_individual_to_raw(self):
        """Helper to populate raw text fields from current individual fields."""
        num_pts = self.num_points_var.get()
        # Ensure individual entries match num_pts before transferring
        if len(self.src_points_entries) != num_pts:
            self._create_point_entry_rows(num_pts)  # Recreate with defaults if mismatch

        current_src_pts_ind = self._get_individual_points(self.src_points_entries)
        if current_src_pts_ind is not None:
            self.src_raw_text.delete("1.0", tk.END)
            self.src_raw_text.insert("1.0", self._format_points_for_raw_text(current_src_pts_ind))

        current_dst_pts_ind = self._get_individual_points(self.dst_points_entries)
        if current_dst_pts_ind is not None:
            self.dst_raw_text.delete("1.0", tk.END)
            self.dst_raw_text.insert("1.0", self._format_points_for_raw_text(current_dst_pts_ind))

    def _update_homography_display(self):
        if self.H is not None:
            for r in range(3):
                for c in range(3):
                    self.homography_matrix_entries[r][c].delete(0, tk.END)
                    self.homography_matrix_entries[r][c].insert(0, f"{self.H[r, c]:.4f}")
        else:
            for r in range(3):
                for c in range(3):
                    self.homography_matrix_entries[r][c].delete(0, tk.END)
                    self.homography_matrix_entries[r][c].insert(0, "N/A")

    def _draw_quad_and_points(self, canvas, points_coords_array, tk_image_bg, color="blue", point_color="red",
                              tag_prefix=""):
        canvas.delete(f"{tag_prefix}_poly");
        canvas.delete(f"{tag_prefix}_points");
        canvas.delete(f"{tag_prefix}_vectors");
        canvas.delete(f"{tag_prefix}_bg")
        canvas.create_image(0, 0, anchor=tk.NW, image=tk_image_bg, tags=f"{tag_prefix}_bg");
        canvas.image = tk_image_bg
        if points_coords_array is None or len(points_coords_array) < 2: return
        num_current_points = len(points_coords_array)
        if num_current_points >= 2:
            flat_points = [coord for pt in points_coords_array for coord in pt]
            if num_current_points >= 3:
                canvas.create_polygon(flat_points, fill="", outline=color, width=self.line_width,
                                      tags=f"{tag_prefix}_poly")
            elif num_current_points == 2:
                canvas.create_line(flat_points[0], flat_points[1], flat_points[2], flat_points[3], fill=color,
                                   width=self.line_width, tags=f"{tag_prefix}_poly")  # Corrected for 2 points
        for x, y in points_coords_array: canvas.create_oval(x - self.point_radius, y - self.point_radius,
                                                            x + self.point_radius, y + self.point_radius,
                                                            fill=point_color, outline=point_color,
                                                            tags=f"{tag_prefix}_points")
        if self.show_vectors_var.get():
            origin = (0, 0)
            if tag_prefix == "src":
                for pt_x, pt_y in points_coords_array: canvas.create_line(origin[0], origin[1], pt_x, pt_y,
                                                                          fill="darkgreen", width=1, arrow=tk.LAST,
                                                                          tags=f"{tag_prefix}_vectors")
            elif tag_prefix == "dst" and self.H is not None:
                current_src_pts_for_vec = self._get_current_src_points()
                if current_src_pts_for_vec is None: return
                origin_h = np.array([0, 0, 1], dtype=np.float32);
                trans_origin_h = self.H @ origin_h
                trans_origin_pt = (trans_origin_h[0] / trans_origin_h[2], trans_origin_h[1] / trans_origin_h[2]) if abs(
                    trans_origin_h[2]) > 1e-6 else (float('inf'), float('inf'))
                for pt_s_x, pt_s_y in current_src_pts_for_vec:
                    pt_s_h = np.array([pt_s_x, pt_s_y, 1], dtype=np.float32);
                    trans_pt_h = self.H @ pt_s_h
                    trans_pt = (trans_pt_h[0] / trans_pt_h[2], trans_pt_h[1] / trans_pt_h[2]) if abs(
                        trans_pt_h[2]) > 1e-6 else (float('inf'), float('inf'))
                    if all(abs(c) < self.canvas_width * 10 for c in trans_origin_pt) and all(
                            abs(c) < self.canvas_height * 10 for c in trans_pt):
                        canvas.create_line(trans_origin_pt[0], trans_origin_pt[1], trans_pt[0], trans_pt[1],
                                           fill="purple", width=1, arrow=tk.LAST, tags=f"{tag_prefix}_vectors")

    def _draw_initial_canvases(self):  # This method is no longer called from __init__
        src_img_tk = ImageTk.PhotoImage(self.base_grid_image_pil);
        self.source_canvas.create_image(0, 0, anchor=tk.NW, image=src_img_tk, tags="src_bg");
        self.source_canvas.image = src_img_tk
        dst_img_tk = ImageTk.PhotoImage(self.base_grid_image_pil.copy());
        self.destination_canvas.create_image(0, 0, anchor=tk.NW, image=dst_img_tk, tags="dst_bg");
        self.destination_canvas.image = dst_img_tk

    def _update_3d_plot(self, src_pts_2d_array, dst_pts_2d_to_plot_array):
        self.ax_3d.clear();
        line_extension_factor = 2.0
        num_current_points = len(src_pts_2d_array) if src_pts_2d_array is not None else 0
        if src_pts_2d_array is not None and dst_pts_2d_to_plot_array is not None and num_current_points >= 3 and len(
                dst_pts_2d_to_plot_array) == num_current_points:
            src_pts_3d = np.concatenate([src_pts_2d_array, np.zeros((num_current_points, 1))], axis=1)
            dst_pts_3d = np.concatenate([dst_pts_2d_to_plot_array, np.ones((num_current_points, 1))], axis=1)
            verts_src = [list(zip(src_pts_3d[:, 0], src_pts_3d[:, 1], src_pts_3d[:, 2]))];
            poly_src = Poly3DCollection(verts_src, alpha=0.6, facecolors='cyan', linewidths=1.5, edgecolors='blue');
            self.ax_3d.add_collection3d(poly_src)
            verts_dst = [list(zip(dst_pts_3d[:, 0], dst_pts_3d[:, 1], dst_pts_3d[:, 2]))];
            poly_dst = Poly3DCollection(verts_dst, alpha=0.6, facecolors='lightgreen', linewidths=1.5,
                                        edgecolors='green');
            self.ax_3d.add_collection3d(poly_dst)
            for i in range(num_current_points):
                s_pt = src_pts_3d[i];
                d_pt = dst_pts_3d[i];
                direction_vector = d_pt - s_pt
                line_start_pt = s_pt - line_extension_factor * direction_vector;
                line_end_pt = d_pt + line_extension_factor * direction_vector
                self.ax_3d.plot([line_start_pt[0], line_end_pt[0]], [line_start_pt[1], line_end_pt[1]],
                                [line_start_pt[2], line_end_pt[2]], 'r--', linewidth=1)
            all_x = np.concatenate([src_pts_3d[:, 0], dst_pts_3d[:, 0]]);
            all_y = np.concatenate([src_pts_3d[:, 1], dst_pts_3d[:, 1]])
            min_x, max_x = (np.min(all_x) if all_x.size > 0 else 0) - self.grid_step, (
                np.max(all_x) if all_x.size > 0 else self.canvas_width) + self.grid_step
            min_y, max_y = (np.min(all_y) if all_y.size > 0 else 0) - self.grid_step, (
                np.max(all_y) if all_y.size > 0 else self.canvas_height) + self.grid_step
            self.ax_3d.set_xlim([min(0, min_x), max(self.canvas_width, max_x)]);
            self.ax_3d.set_ylim([min(0, min_y), max(self.canvas_height, max_y)]);
            self.ax_3d.set_zlim([-3, 3])
            self.ax_3d.set_xlabel('X');
            self.ax_3d.set_ylabel('Y');
            self.ax_3d.set_zlabel('Z (Conceptual Plane)');
            self.ax_3d.set_title('3D Plane-to-Plane Mapping (Extended Rays)')
        else:
            self.ax_3d.set_title('3D Visualization (Need >=3 valid points)');
            self.ax_3d.set_xlabel('X');
            self.ax_3d.set_ylabel('Y');
            self.ax_3d.set_zlabel('Z')
            self.ax_3d.set_xlim([0, self.canvas_width]);
            self.ax_3d.set_ylim([0, self.canvas_height]);
            self.ax_3d.set_zlim([-3, 3])
        self.canvas_3d_widget.draw()

    def _trigger_redraw_visuals(self):
        src_pts = self._get_current_src_points()
        dst_pts_ui = self._get_current_dst_points()

        # Fallback to drawing initial canvases if src_pts are None (e.g. parse error in raw mode)
        if src_pts is None:
            # self._draw_initial_canvases() # This only draws grid, _draw_quad_and_points also does
            # Instead, ensure plain grids are shown by passing None to _draw_quad_and_points
            plain_grid_img_tk = ImageTk.PhotoImage(self.base_grid_image_pil)
            self._draw_quad_and_points(self.source_canvas, None, plain_grid_img_tk, tag_prefix="src")
            self._draw_quad_and_points(self.destination_canvas, None, plain_grid_img_tk, tag_prefix="dst")
            self._update_3d_plot(None, None)
            return

        src_img_tk = ImageTk.PhotoImage(self.base_grid_image_pil)
        self._draw_quad_and_points(self.source_canvas, src_pts, src_img_tk, color="blue", point_color="red",
                                   tag_prefix="src")

        dst_img_tk_bg = ImageTk.PhotoImage(self.base_grid_image_pil.copy())
        if self.H is not None:
            try:
                warped_grid_cv = cv2.warpPerspective(self.base_grid_image_cv, self.H,
                                                     (self.canvas_width, self.canvas_height))
                warped_grid_pil = self._cv2_to_pil(warped_grid_cv)
                dst_img_tk_bg = ImageTk.PhotoImage(warped_grid_pil)
            except cv2.error as e:
                print(f"OpenCV error during warpPerspective: {e}")

        self._draw_quad_and_points(self.destination_canvas, dst_pts_ui, dst_img_tk_bg, color="green",
                                   point_color="orange", tag_prefix="dst")

        self._update_3d_plot(src_pts, dst_pts_ui)

    def _on_compute_homography_from_points(self):
        src_pts = self._get_current_src_points()
        dst_pts_ui = self._get_current_dst_points()

        if src_pts is None or dst_pts_ui is None:
            self.H = None;
            self._update_homography_display();
            self._trigger_redraw_visuals();
            return

        current_num_points = self.num_points_var.get()
        if len(src_pts) < 4 or len(dst_pts_ui) < 4 or current_num_points < 4:
            messagebox.showerror("Input Error",
                                 "At least 4 source and 4 destination points are required to compute Homography.")
            self.H = None;
            self._update_homography_display();
            self._trigger_redraw_visuals();
            return

        if len(src_pts) != len(dst_pts_ui):
            messagebox.showerror("Input Error", "The number of source and destination points must match.")
            self.H = None;
            self._update_homography_display();
            self._trigger_redraw_visuals();
            return

        H_matrix_computed, mask = cv2.findHomography(src_pts, dst_pts_ui, 0)
        self.H = H_matrix_computed if H_matrix_computed is not None else None
        if self.H is None: messagebox.showerror("Error", "Could not compute Homography. Check points.")

        self._update_homography_display();
        self._trigger_redraw_visuals()

    def _on_apply_manual_homography(self):
        manual_H_values = []
        try:
            for r_idx in range(3):
                for c_idx in range(3):
                    manual_H_values.append(float(self.homography_matrix_entries[r_idx][c_idx].get()))
            self.H = np.array(manual_H_values).reshape((3, 3))
        except ValueError:
            messagebox.showerror("Input Error", "Invalid numeric value in Homography Matrix entries.");
            return

        src_pts = self._get_current_src_points()
        if src_pts is None:
            messagebox.showerror("Input Error", "Source points are invalid. Cannot apply manual H.")
            self._trigger_redraw_visuals();
            return

        new_dst_pts_list = []
        for pt_s_coords in src_pts:
            s_h = np.array([pt_s_coords[0], pt_s_coords[1], 1], dtype=np.float32);
            d_h = self.H @ s_h
            new_dst_pts_list.append(
                (d_h[0] / d_h[2], d_h[1] / d_h[2]) if abs(d_h[2]) > 1e-6 else (float('inf'), float('inf')))

        new_dst_pts_np = np.array(new_dst_pts_list, dtype=np.float32)
        self._update_dst_points_display(new_dst_pts_np)

        self._trigger_redraw_visuals()


if __name__ == "__main__":
    root = tk.Tk()
    app = HomographyVisualizerApp(root)
    root.mainloop()