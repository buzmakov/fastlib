# -*- coding: utf-8 -*-
import os
from flask import Flask, render_template, request, send_file, jsonify, send_from_directory
from web_utils import TomoContainer

__author__ = 'makov'

app = Flask(__name__)

#app.config['TOMO_ROOT'] = r'/home/tomo/tomo_data'
app.config['TOMO_ROOT'] = r'/home/makov/tmp/tomo_root/Raw'

tc = TomoContainer()
tc.load_tomo_objects(app.config['TOMO_ROOT'])
app.config['TOMO_CONTAINER'] = tc

@app.route('/')
def index():
    """
    Render start page

    :return:
    """
    return render_template('index.html')


@app.route('/favicon.ico')
def favicon():
    """
    Return favicon

    :return:
    """
    return send_from_directory(os.path.join(app.root_path, 'static', 'ico'),
                               'favicon.ico', mimetype='image/vnd.microsoft.icon')

@app.route('/tomo_experiments/')
def tomo_experiments():
    """
    Render page with list of tomo experiments

    :return:
    """
    app.config['TOMO_CONTAINER'].update_objects_status()
    app.config['TOMO_CONTAINER'].load_tomo_objects(app.config['TOMO_ROOT'])
    tmp_to = [v for (k, v) in app.config['TOMO_CONTAINER'].tomo_objects.items()]
    tmp_to = sorted(tmp_to, key=lambda x: x['config']['description']['date']['start'])
    return render_template('tomo_dirs.html', tomo_objects=tmp_to)


@app.route('/tomo_experiments/info/<tomo_object_id>')
def tomo_info(tomo_object_id):
    """
    Render information page

    :param tomo_object_id:
    :return:
    """
    app.config['TOMO_CONTAINER'].update_objects_status()
    if tomo_object_id in app.config['TOMO_CONTAINER'].tomo_objects:
        to = app.config['TOMO_CONTAINER'].tomo_objects[tomo_object_id]
    else:
        return 'Object NOT found.', 404
    to.load_tomo_config()
    return render_template('tomo_info.html', tomo_object=to)


@app.route('/tomo_experiments/process/<tomo_object_id>')
def tomo_process(tomo_object_id):
    """
    Render page for tomograthic processing

    :param tomo_object_id:
    :return:
    """
    app.config['TOMO_CONTAINER'].update_objects_status()
    if tomo_object_id in app.config['TOMO_CONTAINER'].tomo_objects:
        to = app.config['TOMO_CONTAINER'].tomo_objects[tomo_object_id]
    else:
        return 'Object NOT found.', 404
    return render_template('tomo_process.html', tomo_object=to)


@app.route('/tomo_experiments/results/<tomo_object_id>')
def tomo_result(tomo_object_id):
    """
    Render page for tomograthic results

    :param tomo_object_id:
    :return:
    """
    app.config['TOMO_CONTAINER'].update_objects_status()
    if tomo_object_id in app.config['TOMO_CONTAINER'].tomo_objects:
        to = app.config['TOMO_CONTAINER'].tomo_objects[tomo_object_id]
    else:
        return 'Object NOT found.', 404
    return render_template('tomo_results.html', tomo_object=to)


@app.route('/tomo_experiments/file/<tomo_object_id>')
def get_result_file(tomo_object_id):
    """
    Get result file by tomo_object_id. File name passed in 'fname' argument of HTTP-request.

    :param tomo_object_id:
    :return:
    """
    filename = request.args.get('fname')

    if tomo_object_id in app.config['TOMO_CONTAINER'].tomo_objects:
        to = app.config['TOMO_CONTAINER'].tomo_objects[tomo_object_id]
    else:
        return 'Object NOT found.', 404

    file_name = os.path.join(to.get_full_path(), filename)
    if not to.is_path_in_datapath(filename):
        return 'File not found', 404

    if os.path.isfile(file_name):
        return send_file(file_name, as_attachment=True)
    else:
        return 'File not found', 404


@app.route('/tomo_experiments/log/<tomo_object_id>')
def tomo_log(tomo_object_id):
    """
    Return tomographi reconstruction log. Used in AJAX-mode.

    :param tomo_object_id:
    :return:
    """
    if tomo_object_id in app.config['TOMO_CONTAINER'].tomo_objects:
        to = app.config['TOMO_CONTAINER'].tomo_objects[tomo_object_id]
        return jsonify(log=to.get_log())
    else:
        return 'Object NOT found.', 404


@app.route('/tomo_experiments/reconstruct/<tomo_object_id>')
def tomo_reconstruct(tomo_object_id):
    """
    Start tomo reconstruction or image processing.

    :param tomo_object_id:
    :return:
    """
    just_preprocess = False
    if 'just_preprocess' in request.args:
        if request.args['just_preprocess'] == 'True':
            just_preprocess = True

    app.config['TOMO_CONTAINER'].add_to_reconstruction_queue(tomo_object_id, just_preprocess=just_preprocess)
    return jsonify(status='OK. just_preprocess == ' + str(just_preprocess))

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
