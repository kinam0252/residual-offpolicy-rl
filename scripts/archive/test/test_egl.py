import os
os.environ['MUJOCO_GL'] = 'egl'
os.environ['PYOPENGL_PLATFORM'] = 'egl'
import mujoco
print('mujoco OK')
xml = '<mujoco><worldbody><light pos="0 0 1"/><geom type="sphere" size="0.1"/></worldbody></mujoco>'
m = mujoco.MjModel.from_xml_string(xml)
r = mujoco.Renderer(m, height=64, width=64)
d = mujoco.MjData(m)
mujoco.mj_forward(m, d)
r.update_scene(d)
frame = r.render()
print(f'frame: shape={frame.shape}, min={frame.min()}, max={frame.max()}')
