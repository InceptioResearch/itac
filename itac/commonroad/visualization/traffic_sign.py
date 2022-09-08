import copy
import enum
import os
from collections import defaultdict
from typing import Dict
from scipy.cluster.hierarchy import linkage, fcluster

from PIL import Image
from commonroad.geometry.shape import *
from commonroad.scenario.traffic_sign import TrafficSign, \
    TrafficLight, \
    TrafficSignIDUsa
from commonroad.visualization.param_server import ParamServer
from matplotlib.axes import Axes, mtext
from matplotlib.offsetbox import OffsetImage, \
    AnnotationBbox, \
    HPacker, \
    TextArea, \
    VPacker, \
    OffsetBox

# path to traffic sign .png files
traffic_sign_path = os.path.join(os.path.dirname(__file__), 'traffic_signs/')

speed_limit_factors = {'mph': 2.23694, 'kmh': 3.6, 'ms': 1.0}

# default scaling for traffic sign images and annotations
px_per_metre = 0.018
px_per_metre_text = 0.078


class TextAreaAutoscale(TextArea):
    def __init__(self, s,
                 px_per_metre=px_per_metre_text,
                 textprops=None,
                 multilinebaseline=None):
        super().__init__(s,
                         textprops=textprops,
                         multilinebaseline=multilinebaseline)

        self.px_per_metre = px_per_metre
        textprops_init = copy.copy(textprops)
        if textprops_init is None:
            textprops_init = {}
        textprops_init.setdefault("va", "center")
        textprops_init.setdefault("ha", "center")
        self.textprops_init = textprops_init
        self.dx_m = None
        self.dy_m = None
        self.dx_pix = None
        self.dy_pix = None

    def set_ax_lims(self, dx_m, dy_m, dx_pix, dy_pix):
        self.dx_m = dx_m
        self.dy_m = dy_m
        self.dx_pix = dx_pix
        self.dy_pix = dy_pix

    def ax_update(self, ax: Axes):
        '''
        Update image size based on axes limits and window size in pixels
        '''
        bbox = ax.get_window_extent()
        width_px, height_px = bbox.width, bbox.height
        # Get the range for the new area
        _, _, xdelta, ydelta = ax.viewLim.bounds
        self.set_ax_lims(xdelta, ydelta, width_px, height_px)

    def get_extent(self, renderer):
        correction_x = self.dx_pix / self.dx_m * self.px_per_metre
        correction_y = self.dy_pix / self.dy_m * self.px_per_metre
        corr = min(correction_x, correction_y)
        if 'size' in self.textprops_init:
            textprops = copy.copy(self.textprops_init)
            textprops['size'] *= corr
            f = self._text.figure
            self._text = mtext.Text(0, 0, self.get_text(), **textprops)
            self._text.figure = f
            self._children = [self._text]
            self._text.set_transform(self.offset_transform +
                                     self._baseline_transform)
        _, h_, d_ = renderer.get_text_width_height_descent(
            "lp", self._text._fontproperties, ismath=False)

        bbox, info, d = self._text._get_layout(renderer)
        w, h = bbox.width, bbox.height

        self._baseline_transform.clear()

        if len(info) > 1 and self._multilinebaseline:
            d_new = 0.5 * h - 0.5 * (h_ - d_)
            self._baseline_transform.translate(0, d - d_new)
            d = d_new

        else:  # single line
            h_d = max(h_ - d_, h - d)
            d = max(d, d_)
            h = h_d + d

        return w * corr, h * corr, 0., d * corr


class OffsetImageAutoscale(OffsetImage):
    """OffsetImage that scales proportionally to data units (not display units, i.e. pixels)
    and includes overlayed text."""

    def __init__(self, arr,
                 px_per_metre=px_per_metre,
                 px_per_metre_text=px_per_metre_text,
                 text=None,
                 txt_offset_x=0.7,
                 txt_offset_y=0.45,
                 textprops=None,
                 cmap=None,
                 norm=None,
                 interpolation=None,
                 origin=None,
                 filternorm=True,
                 filterrad=4.0,
                 resample=False,
                 dpi_cor=True,
                 **kwargs
                 ):
        super().__init__(arr,
                         cmap=cmap,
                         norm=norm,
                         interpolation=interpolation,
                         origin=origin,
                         filternorm=filternorm,
                         filterrad=filterrad,
                         resample=resample,
                         dpi_cor=dpi_cor,
                         **kwargs)

        self.text_area = TextAreaAutoscale(text, px_per_metre=px_per_metre_text, textprops=textprops)
        self.txt_offset_x = txt_offset_x
        self.txt_offset_y = txt_offset_y
        self.px_per_metre = px_per_metre
        self.dx_m = None
        self.dy_m = None
        self.dx_pix = None
        self.dy_pix = None

    def set_ax_lims(self, dx_m, dy_m, dx_pix, dy_pix):
        self.dx_m = dx_m
        self.dy_m = dy_m
        self.dx_pix = dx_pix
        self.dy_pix = dy_pix

    def ax_update(self, ax: Axes):
        '''
        Update image size based on axes limits and window size in pixels
        '''
        bbox = ax.get_window_extent()
        width_px, height_px = bbox.width, bbox.height
        # Get the range for the new area
        _, _, xdelta, ydelta = ax.viewLim.bounds
        self.set_ax_lims(xdelta, ydelta, width_px, height_px)
        self.text_area.set_ax_lims(xdelta, ydelta, width_px, height_px)

    def get_extent(self, renderer=None):
        self.text_area._text.figure = self.figure
        if renderer is not None:
            self.text_area.get_extent(renderer)

        if renderer is not None and self._dpi_cor:  # True, do correction
            dpi_cor = renderer.points_to_pixels(1.)
        else:
            dpi_cor = 1.

        data = self.get_data()
        correction_x = self.dx_pix / self.dx_m * self.px_per_metre
        correction_y = self.dy_pix / self.dy_m * self.px_per_metre
        corr = min(correction_x, correction_y)
        ny, nx = data.shape[:2]
        w, h = dpi_cor * nx * corr, dpi_cor * ny * corr

        return w, h, 0, 0

    def draw(self, renderer):
        # docstring inherited
        self.image.draw(renderer)
        self.text_area._text.figure = self.figure
        self.text_area._text.draw(renderer)
        self.stale = False

    def set_offset(self, xy):
        super().set_offset(xy)
        self.text_area.offset_transform.clear()
        w, h, _, _ = self.get_extent()
        xy = (xy[0] + w * self.txt_offset_x, xy[1] + h * self.txt_offset_y)
        self.text_area.offset_transform.translate(xy[0], xy[1])


def isfloat(value: str):
    try:
        float(value)
        return True
    except ValueError:
        return False


def speed_limit_factor(country_code) -> float:
    """Determine factor for speed_limit_unit by country code."""
    # dicts for units other than kph
    mph_countries = [TrafficSignIDUsa]

    if type(country_code) in mph_countries:
        return speed_limit_factors['mph']
    else:
        return speed_limit_factors['kmh']


# denotes traffic signs that are speed limits
unit_conversion_required = ['274', '274.1', '275', '1004-31', 'R2-1', 'r301']


def text_prop_dict() -> dict:
    """Properties of text for additional_value.
    mpl_args: text properties
    rescale_threshold: max. length of additional value string. Above this length, the font size is scaled down
    position_offset_y: vertical offset of additional text measured from the bottom of the traffic sign
        (proportional to traffic sign size)
    """
    return {
        'default': {
            'mpl_args': {
                'weight': 'bold', 'size': 13.5
            },
            'rescale_threshold': 2,
            'position_offset_y': 0.45
        },
        '262': {
            'mpl_args': {
                'weight': 'bold', 'size': 8
            },
            'rescale_threshold': 2,
            'position_offset_y': 0.53
        },
        '264': {
            'mpl_args': {
                'weight': 'bold', 'size': 12
            },
            'rescale_threshold': 2,
            'position_offset_y': 0.47
        },
        '265': {
            'mpl_args': {
                'weight': 'normal', 'size': 13
            },
            'rescale_threshold': 2,
            'position_offset_y': 0.45
        },
        '266': {
            'mpl_args': {
                'weight': 'bold', 'size': 5
            },
            'rescale_threshold': 2,
            'position_offset_y': 0.37
        },
        '274': {
            'mpl_args': {
                'weight': 'bold', 'size': 13.5
            },
            'rescale_threshold': 2,
            'position_offset_y': 0.45
        },
        '274.1': {
            'mpl_args': {
                'weight': 'bold', 'size': 8.5
            },
            'rescale_threshold': 2,
            'position_offset_y': 0.65
        },
        '275': {
            'mpl_args': {
                'weight': 'bold', 'color': 'white', 'size': 13.5
            }, 'rescale_threshold': 2,
            'position_offset_y': 0.45
        },
        '278': {
            'mpl_args': {
                'weight': 'bold', 'color': 'gray', 'size': 10
            },
            'position_offset': -16.5,
            'position_offset_y': 0.45
        },
        '279': {
            'mpl_args': {
                'weight': 'bold', 'color': 'white', 'size': 10
            },
            'position_offset': -16.5,
            'position_offset_y': 0.45
        },
        '310': {
            'mpl_args': {
                'weight': 'normal', 'color': 'black', 'size': 10,
            },
            'position_offset_y': -0.55
        },
        '380': {
            'mpl_args': {
                'weight': 'bold', 'color': 'white', 'size': 10
            },
            'position_offset_y': 0.45
        },
        '381': {
            'mpl_args': {
                'weight': 'bold', 'color': 'white', 'size': 10
            },
            'position_offset_y': 0.45
        },
        '1040-30': {
            'mpl_args': {
                'weight': 'normal', 'color': 'black', 'size': 8
            },
            'position_offset_y': 0.4
        },
        '1001-30': {
            'mpl_args': {
                'weight': 'normal', 'color': 'black', 'size': 8
            },
            'position_offset_y': 0.4
        },
        '1004-31': {
            'mpl_args': {
                'weight': 'normal', 'color': 'black', 'size': 8
            },
            'position_offset_y': 0.4
        },
        'R2-1': {
            'mpl_args': {
                'weight': 'normal', 'color': 'black', 'size': 10.5
            },
            'position_offset_y': 0.3
        },
        'r301': {
            'mpl_args': {
                'weight': 'bold', 'size': 13.5
            },
            'rescale_threshold': 2,
            'position_offset_y': 0.45
        },
    }


def rescale_text(string: str, prop: dict, scale_factor: float,
                 default_scale_factor: float) -> dict:
    """Rescales text size proportionally to the max. number of strings given
    by prop['rescale_threshold'] and to the
    'scale_factor' compared to the default scale_factor. Used e.g. for
    fitting speed limits into the traffic sign."""
    prop = copy.deepcopy(prop)
    if default_scale_factor != scale_factor:
        tmp_scale_factor = scale_factor / default_scale_factor
        if 'position_offset' in prop:
            prop['position_offset'] *= tmp_scale_factor

        if 'mpl_args' in prop and 'size' in prop['mpl_args']:
            prop['mpl_args']['size'] *= tmp_scale_factor

    if 'rescale_threshold' in prop:
        if len(string) > prop['rescale_threshold']:
            if 'mpl_args' in prop and 'size' in prop['mpl_args']:
                prop['mpl_args']['size'] *= prop['rescale_threshold'] / len(
                        string) * 1.1
            if 'position_offset' in prop:
                prop['position_offset'] *= prop['rescale_threshold'] / len(
                        string) * 1.35

    return prop


def create_img_boxes_traffic_sign(
        traffic_signs: Union[List[TrafficSign], TrafficSign],
        draw_params: ParamServer, call_stack: Tuple[str, ...], rnd) -> Dict[Tuple[float, float], List[OffsetBox]]:
    """
    For each Traffic sign an OffsetBox is created, containing the png image
    and optionally labels. These boxes can
    be stacked horizontally later when multiple signs or lights share the
    same position.
    :param traffic_signs:
    :param draw_params:
    :param call_stack:
    :param rnd: MPRenderer
    :return:
    """
    if type(traffic_signs) is not list:
        traffic_signs = [traffic_signs]

    if len(traffic_signs) == 0:
        return dict()

    scale_factor = draw_params.by_callstack(call_stack,
                                            ('traffic_sign', 'scale_factor'))
    speed_limit_unit = draw_params.by_callstack(call_stack, (
            'traffic_sign', 'speed_limit_unit'))
    show_label_default = draw_params.by_callstack(call_stack, (
            'traffic_sign', 'show_label'))
    show_traffic_signs = draw_params.by_callstack(call_stack, (
            'traffic_sign', 'show_traffic_signs'))
    zorder = draw_params.by_callstack(call_stack, ('traffic_sign', 'zorder'))

    scale_factor_default = draw_params.by_callstack(call_stack, (
            'traffic_sign', 'scale_factor'))

    assert any([show_traffic_signs == 'all',
                isinstance(show_traffic_signs, list) and type(
                        show_traffic_signs[0] is enum)]), 'Plotting option ' \
                                                          'traffic_sign.show_traffic_signs must ' \
                                                          'be either "all" or ' \
                                                          '' \
                                                          '' \
                                                          'list of type ' \
                                                          'TrafficSignID'

    prop_dict = text_prop_dict()
    imageboxes_all = defaultdict(list)

    for traffic_sign in traffic_signs:
        if traffic_sign.virtual is True or traffic_sign.position is None:
            continue
        imageboxes = []
        for element in traffic_sign.traffic_sign_elements:
            el_id = element.traffic_sign_element_id
            if not (show_traffic_signs == 'all' or el_id in show_traffic_signs):
                continue
            show_label = show_label_default
            path = os.path.join(traffic_sign_path, el_id.__class__.__name__,
                                el_id.value + '.png')
            plot_img = True
            # get png image
            if not os.path.exists(path):
                warnings.warn(f"File for traffic sign {element} at {path} does not exist!")
                path = os.path.join(traffic_sign_path, '.png')
                if not os.path.exists(path):
                    show_label = True
                    plot_img = False

            boxes = []  # collect matplotlib offset boxes for text and images
            if show_label:
                boxes.append(TextArea(el_id.value))

            if plot_img:
                # plot traffic sign
                sign_img = Image.open(path)
                if len(element.additional_values) > 0:
                    if element.traffic_sign_element_id.value in unit_conversion_required \
                            and isfloat(element.additional_values[0]):
                        if speed_limit_unit == 'auto':
                            speed_factor = speed_limit_factor(
                                element.traffic_sign_element_id)
                        else:
                            speed_factor = speed_limit_factors[speed_limit_unit]

                        add_text = str(round(
                            speed_factor * float(element.additional_values[0])))
                    else:
                        add_text = '\n'.join(element.additional_values)

                    props = prop_dict[el_id.value] if el_id.value in prop_dict else prop_dict['default']
                    props = rescale_text(add_text, props, scale_factor,
                                         scale_factor_default)
                    props_txt = props['mpl_args']
                    txt_offset_y = props['position_offset_y'] if 'position_offset_y' in props else -0.2
                else:
                    txt_offset_y = -0.2
                    props_txt = None
                    add_text = None

                if sign_img.mode != "RGBA":
                    sign_img = sign_img.convert("RGBA")

                boxes.append(
                    OffsetImageAutoscale(sign_img, text=add_text,
                                         px_per_metre=px_per_metre*scale_factor,
                                         px_per_metre_text=px_per_metre_text*scale_factor,
                                         txt_offset_y=txt_offset_y,
                                         textprops=props_txt,
                                         zorder=zorder,
                                         resample=True))
                # add callback for automatic rescaling of image
                rnd.add_callback('xlim_changed', boxes[-1].ax_update)

            # already stack label and img in case both are shown (prevents
            # misalignment with additional text)
            if len(boxes) > 1:
                boxes = [VPacker(children=boxes, pad=0, sep=0, align='center')]

            sep = 0
            # get additional values string (like speed limits) in case the image is not plotted
            if not plot_img:
                if element.traffic_sign_element_id.value in unit_conversion_required \
                        and isfloat(
                        element.additional_values[0]):
                    if speed_limit_unit == 'auto':
                        speed_factor = speed_limit_factor(
                                element.traffic_sign_element_id)
                    else:
                        speed_factor = speed_limit_factors[speed_limit_unit]

                    add_text = str(round(
                            speed_factor * float(element.additional_values[0])))
                else:
                    add_text = '\n'.join(element.additional_values)

                props = prop_dict[
                    el_id.value] if el_id.value in prop_dict else {
                        'mpl_args': {}
                }
                props = rescale_text(add_text, props, scale_factor,
                                     scale_factor_default)
                boxes.append(TextAreaAutoscale(add_text, px_per_metre=px_per_metre_text * scale_factor,
                                               textprops=props['mpl_args']))
                # add callback for automatic rescaling of text
                rnd.add_callback('xlim_changed', boxes[-1].ax_update)

            # stack boxes vertically
            img = VPacker(children=boxes, pad=0, sep=sep, align='center')
            imageboxes.append(img)

        # horizontally stack all traffic sign elements of the traffic sign
        if len(imageboxes) > 0:
            hbox = HPacker(children=imageboxes, pad=0, sep=0.05,
                           align='baseline')
            imageboxes_all[tuple(traffic_sign.position.tolist())].append(hbox)

    return imageboxes_all


def create_img_boxes_traffic_lights(
        traffic_lights: Union[List[TrafficLight], TrafficLight],
        draw_params: ParamServer, call_stack: Tuple[str, ...], rnd) -> Dict[Tuple[float, float], List[OffsetBox]]:
    """
    For each Traffic light an OffsetBox is created, containing the png image
    and optionally labels. These boxes can
    be stacked horizontally later when multiple signs or lights share the
    same position.
    :param traffic_lights:
    :param draw_params:
    :param call_stack:
    :return:
    """
    if type(traffic_lights) is not list:
        traffic_lights = [traffic_lights]

    if len(traffic_lights) == 0:
        return dict()

    time_begin = draw_params.by_callstack(call_stack, ('time_begin',))
    scale_factor = draw_params.by_callstack(call_stack,
                                            ('traffic_light', 'scale_factor'))
    show_label = draw_params.by_callstack(call_stack,
                                          ('traffic_light', 'show_label'))
    zorder = draw_params.by_callstack(call_stack, ('traffic_light', 'zorder'))

    # plots all group members horizontally stacked
    imageboxes_all = defaultdict(list)
    for traffic_light in traffic_lights:
        if traffic_light.position is None:
            continue
        if traffic_light.active:
            state = traffic_light.get_state_at_time_step(time_begin)
            path = os.path.join(traffic_sign_path, 'traffic_light_state_' + str(
                    state.value) + '.png')
        else:
            path = os.path.join(traffic_sign_path,
                                'traffic_light_state_inactive.png')

        boxes = []  # collect matplotlib offset boxes for text and images
        sign_img = Image.open(path)
        boxes.append(OffsetImageAutoscale(sign_img,
                                          px_per_metre=px_per_metre * scale_factor,
                                          zorder=zorder,
                                          resample=True))
        rnd.add_callback('xlim_changed', boxes[-1].ax_update)

        if show_label:
            boxes.append(TextArea(str(state.value)))

        # stack boxes vertically
        img_box = VPacker(children=boxes, pad=0, sep=0, align='center')

        imageboxes_all[tuple(traffic_light.position.tolist())].append(img_box)

    return imageboxes_all


def draw_traffic_light_signs(traffic_lights_signs: Union[
    List[Union[TrafficLight, TrafficSign]], TrafficLight, TrafficSign],
                             draw_params: ParamServer,
                             call_stack: Tuple[str, ...],
                             rnd):
    """
    Draws OffsetBoxes which are first collected for all traffic signs and
    -lights. Boxes are stacked together when they
    share the same position.
    :param traffic_lights_signs:
    :param plot_limits:
    :param ax:
    :param draw_params:
    :param draw_func:
    :param handles:
    :param call_stack:
    :param rnd: MPRenderer
    :return:
    """
    kwargs = draw_params.by_callstack(call_stack, ('lanelet_network', 'kwargs_traffic_light_signs'))

    zorder_0 = draw_params.by_callstack(call_stack, ('traffic_light', 'zorder'))

    zorder_1 = draw_params.by_callstack(call_stack, ('traffic_sign', 'zorder'))

    zorder = min(zorder_0, zorder_1)
    threshold_grouping = 0.8  # [m] distance threshold for grouping traffic
    # light and/or signs

    if not isinstance(traffic_lights_signs, list):
        traffic_lights_signs = [traffic_lights_signs]

    traffic_signs = []
    traffic_lights = []

    for obj in traffic_lights_signs:
        if isinstance(obj, TrafficSign):
            traffic_signs.append(obj)
        elif isinstance(obj, TrafficLight):
            traffic_lights.append(obj)
        else:
            warnings.warn('Object of type {}, but expected type TrafficSign or '
                          'TrafficLight'.format(type(obj)))

    # collect ImageBoxes of traffic signs/lights grouped by their positions
    boxes_tl = create_img_boxes_traffic_lights(traffic_lights, draw_params,
                                               call_stack, rnd)
    boxes_signs = create_img_boxes_traffic_sign(traffic_signs, draw_params,
                                                call_stack, rnd)

    img_boxes = defaultdict(list)  # {position: List[OffsetBox]}
    [img_boxes[pos].extend(box_list) for pos, box_list in boxes_tl.items()]
    [img_boxes[pos].extend(box_list) for pos, box_list in boxes_signs.items()]

    if not img_boxes:
        return []

    positions = list(img_boxes.keys())
    box_lists = list(img_boxes.values())

    # group objects based on their positions' distances
    group_boxes = defaultdict(list)
    group_positions = defaultdict(list)
    groups = defaultdict(list)

    # find clusters where minimal pairwise distance is below distance threshold_grouping
    if len(positions) <= 1:
        groups[positions[0]] = box_lists[0]
    else:
        Z = linkage(np.array(positions), 'single', metric='chebyshev')
        clusters = fcluster(Z, threshold_grouping, criterion='distance')
        for i, cluster_id in enumerate(clusters):
            group_boxes[cluster_id].extend(box_lists[i])
            group_positions[cluster_id].append(positions[i])

        for cluster_id, boxes in group_boxes.items():
            groups[tuple(np.average(group_positions[cluster_id], axis=0).tolist())] = boxes

    # add default AnnotationBox args if not specified by user
    default_params = dict(xycoords='data', frameon=False)
    for param, value in default_params.items():
        if param not in kwargs:
            kwargs[param] = value

    artists = []
    # stack imageboxes of each group and draw
    for position_tmp, box_list_tmp in groups.items():
        position_tmp = np.array(position_tmp)
        kwargs_tmp = copy.deepcopy(kwargs)
        if 'xybox' not in kwargs_tmp:
            kwargs_tmp['xybox'] = position_tmp

        hbox = HPacker(children=box_list_tmp, pad=0, sep=0.1, align='baseline')
        ab = AnnotationBbox(hbox, position_tmp, **kwargs_tmp)
        ab.zorder = zorder
        artists.append(ab)
    return artists
