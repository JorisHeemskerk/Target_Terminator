"""
Target yaml template.

This template can be used to validate the composition of any custom
target yaml file. This is also already done when creating an actual
environment.

Minimal viable yaml file looks like (_'s are placeholders for data):

```yaml
size : [_, _]
position : [_, _]
coll_radius : _
```
"""

TARGET_TEMPLATE = {
    'sprite' : {
        'required' : False,
        'type' : 'string'
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
    'position' : {
        'required' : True,
        'type' : 'list',
        'minlength' : 2, 
        'maxlength' : 2,
        'items' : [
            {'type' : 'integer', 'min' : 0}, 
            {'type' : 'integer', 'min' : 0}
        ]
    },
    'coll_radius' : {
        'required' : True,
        'type' : 'number',
        'min' : 0
    }
}
