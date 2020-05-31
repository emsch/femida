import jinja2
import os


if __name__ == '__main__':
    with open('/etc/nginx/conf.d/nginx.conf.template') as f:
        template = jinja2.Template(f.read())
    result = template.render(**os.environ)
    with open('/etc/nginx/conf.d/nginx.conf', 'w') as f:
        f.write(result)
