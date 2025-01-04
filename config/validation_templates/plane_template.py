"""
Plane yaml template.

This template can be used to validate the composition of any custom
plane yaml file. This is also already done when creating an actual
environment.

Minimal viable yaml file looks like (_'s are placeholders for data):

```yaml
properties:
    mass : _
    drag_constant : _
    engine_force : _
    agility : _
    lift_constant : _
    critical_aoa_lower_bound : [_, _] 
    critical_aoa_higher_bound : [_, _]
    lift_coefficient_aoa_0 : _
    drag_coefficient_aoa_0 : _
starting_config:
    initial_throttle : _
    initial_pitch : _
    initial_velocity : [_, _]
    initial_position : [_, _]
    position_px_deviation : _
    size : [_, _] 
bullet_config:
    speed : _
    lifetime : _
    size : [_, _]
```
"""

PLANE_TEMPLATE = {
    'sprite' : {
        'required' : False,
        'type' : 'dict',
        'schema' : {
            'side_view_dir' : {
                'required' : False,
                'type' : 'string'
            },
            'top_view_dir' : {
                'required' : False,
                'type' : 'string'
            },
            'size' : {
                'required' : True,
                'type' : 'list',
                'minlength' : 2, 
                'maxlength' : 2,
                'items' : [
                    {'type' : 'integer', 'min' : 0}, 
                    {'type': 'integer', 'min' : 0}
                ]
            },
        }
    },
    'properties' : {
        'required' : True,
        'type' : 'dict',
        'schema' : {
            # scalars:
            'mass' : {
                'type' : 'number',
                'min' : 0
            },
            'drag_constant' : {
                'type' : 'number',
                'min' : 0
            },
            'lift_constant' : {
                'type' : 'number',
                'min' : 0
            },
            'lift_coefficient_aoa_0' : {
                'type' : 'number',
                'min' : 0
            },
            'drag_coefficient_aoa_0' : {
                'type' : 'number',
                'min' : 0
            },
            'engine_force' : {
                'type' : 'number',
                'min' : 0
            },
            'agility' : {
                'type' : 'number',
                'min' : 0
            },
            'initial_throttle' : {
                'type' : 'number',
                'min' : 0
            },
            'initial_pitch' : {
                'type' : 'number',
                'min' : 0
            },
            'collision_radius' : {
                'type' : 'number',
                'min' : 0
            },
            # Vectors:
            'critical_aoa_lower_bound' : {
                'required' : True,
                'type' : 'list',
                'minlength' : 2, 
                'maxlength' : 2,
                'items' : [
                    {'type' : 'number'}, 
                    {'type': 'number'}
                ]
            },
            'critical_aoa_higher_bound' : {
                'required' : True,
                'type' : 'list',
                'minlength' : 2, 
                'maxlength' : 2,
                'items' : [
                    {'type' : 'number'}, 
                    {'type': 'number'}
                ]
            },
            'initial_velocity' : {
                'required' : True,
                'type' : 'list',
                'minlength' : 2, 
                'maxlength' : 2,
                'items' : [
                    {'type' : 'number'}, 
                    {'type': 'number'}
                ]
            },
            'initial_position' : {
                'required' : True,
                'type' : 'list',
                'minlength' : 2, 
                'maxlength' : 2,
                'items' : [
                    {'type' : 'integer', 'min' : 0}, 
                    {'type': 'integer', 'min' : 0}
                ]
            }
        }
    },
    'bullet_config' : {
        'required' : True,
        'type' : 'dict',
        'schema' : {
            'sprite' : {
                'required' : False,
                'type' : 'string'
            },
            'speed' : {
                'required' : True,
                'type' : 'number',
                'min' : 0
            },
            'lifetime' : {
                'required' : True,
                'type' : 'number',
                'min' : 0
            },
            'size' : {
                'required' : True,
                'type' : 'list',
                'minlength' : 2, 
                'maxlength' : 2,
                'items' : [
                    {'type' : 'integer', 'min' : 1}, 
                    {'type': 'integer', 'min' : 1}
                ]
            },
        }
    }
}
