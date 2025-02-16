import ee

def authenticate_and_initialise():
    try:
        ee.Authenticate()
        ee.Initialize(project='ee-nicolasrenatoarroyo')
        print(ee.String('Hello from the Earth Engine servers!').getInfo())
    except:
        print('Failed connection to Earth Engine servers')