import ctypes
import os


def gsp_api_initialize(engine_model):
    # Step 1: Add the DLL directory to the system path
    dll_path = r"C:\GSP_thesis"  # Replace with the actual path


    # Step 2: Load the DLL
    gspdll = ctypes.CDLL(os.path.join(dll_path, 'GSP.dll'))

    # FreeAll - no arguments, returns bool
    gspdll.FreeAll.argtypes = []
    gspdll.FreeAll.restype = ctypes.c_bool

    # CloseModel - takes a bool, returns bool
    gspdll.CloseModel.argtypes = [ctypes.c_bool]
    gspdll.CloseModel.restype = ctypes.c_bool

    # LoadModelAnsi - takes char* and two bools, returns bool
    gspdll.LoadModelAnsi.argtypes = [ctypes.c_char_p, ctypes.c_bool, ctypes.c_bool]
    gspdll.LoadModelAnsi.restype = ctypes.c_bool

    # RunModel - takes four bools, returns bool
    gspdll.RunModel.argtypes = [ctypes.c_bool, ctypes.c_bool, ctypes.c_bool, ctypes.c_bool]
    gspdll.RunModel.restype = ctypes.c_bool

    # CalculateDesignPoint - takes three bools, returns bool
    gspdll.CalculateDesignPoint.argtypes = [ctypes.c_bool, ctypes.c_bool, ctypes.c_bool]
    gspdll.CalculateDesignPoint.restype = ctypes.c_bool

    # SetInputControlParameterByIndex - takes an int and a double, returns bool
    gspdll.SetInputControlParameterByIndex.argtypes = [ctypes.c_int, ctypes.c_double]
    gspdll.SetInputControlParameterByIndex.restype = ctypes.c_bool

    # GetOutputDataParameterValueByIndex - takes an int, double pointer, and bool, returns bool
    gspdll.GetOutputDataParameterValueByIndex.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_double), ctypes.c_bool]
    gspdll.GetOutputDataParameterValueByIndex.restype = ctypes.c_bool



    # Load a model using LoadModelAnsi
    if engine_model == 'GTF':
        model_path = r"C:\GSP_thesis\GTF_from_scratch_GSP12_V17_detoriation_26_6.mxl"
    elif engine_model == 'GTF2035':
        model_path = r"C:\GSP_thesis\GTF2035_V1.mxl"
    elif engine_model == 'GTF2035_wi':
        model_path = r"C:\GSP_thesis\GTF2035_wi_V2.mxl"
    elif engine_model == 'GTF1990':
        model_path =  r"C:\GSP_thesis\CFM56_5B4_P_v3.mxl"  #this is actually GTF2000 / 2008 gsp model, but performance very very similar
    else:
        print('Not a correct engine model name')
        model_path = None

    if gspdll.LoadModelAnsi(model_path.encode('utf-8'), False, False):
        print("Model loaded successfully.")
    else:
        print("Failed to load the model.")
    print('model config',gspdll.ConfigureModel())


    # Step 4: Call 'CalculateDesignPoint' with arguments (0, 0, 1)
    result = gspdll.CalculateDesignPoint(False, False, False)  # Python `False` is equivalent to 0 and `True` is 1

    if result:
        print("CalculateDesignPoint executed successfully.")
    else:
        print("CalculateDesignPoint execution failed.")

    return gspdll


def gsp_api_close(gspdll):
    # Close the model
    if gspdll.CloseModel(True):
        print("Model closed successfully.")
    else:
        print("Failed to close the model.")

    # Free all allocated resources
    if gspdll.FreeAll():
        print("Resources freed successfully.")
    else:
        print("Failed to free resources.")

    # Step 5: Unload the DLL to release it from memory
    # Get the handle to the DLL and free it using FreeLibrary
    dll_handle = gspdll._handle  # Access the handle of the loaded DLL
    ctypes.windll.kernel32.FreeLibrary(dll_handle)
    #
    # # Remove the reference to the DLL
    del gspdll
    print("DLL unloaded successfully.")

    return


def process_single_row_direct(gspdll, mach, specific_humidity, air_temperature, air_pressure, thrust_per_engine, water_injection_kg_s, lhv, engine, flight_phase):
    try:
        # Set input parameters directly
        gspdll.SetInputControlParameterByIndex(1, mach)
        gspdll.SetInputControlParameterByIndex(2, specific_humidity*100)
        gspdll.SetInputControlParameterByIndex(3, air_temperature)
        gspdll.SetInputControlParameterByIndex(4, air_pressure)
        gspdll.SetInputControlParameterByIndex(5, thrust_per_engine)
        gspdll.SetInputControlParameterByIndex(6, water_injection_kg_s)
        gspdll.SetInputControlParameterByIndex(7, lhv)
        if (engine == 'GTF' or engine == 'GTF2035' or engine == 'GTF2035_wi') and flight_phase == 'cruise':
            gspdll.SetInputControlParameterByIndex(8, -2.0)
            gspdll.SetInputControlParameterByIndex(9, -2.0)
            gspdll.SetInputControlParameterByIndex(10, -2.0)
            gspdll.SetInputControlParameterByIndex(11, -2.0)
            gspdll.SetInputControlParameterByIndex(12, -2.0)
            gspdll.SetInputControlParameterByIndex(13, -2.0)
            print('-2% deteoriation')
        elif (engine == 'GTF' or engine == 'GTF2035' or engine == 'GTF2035_wi') and flight_phase != 'cruise':
            gspdll.SetInputControlParameterByIndex(8, 0.0)
            gspdll.SetInputControlParameterByIndex(9, 0.0)
            gspdll.SetInputControlParameterByIndex(10, 0.0)
            gspdll.SetInputControlParameterByIndex(11, 0.0)
            gspdll.SetInputControlParameterByIndex(12, 0.0)
            gspdll.SetInputControlParameterByIndex(13, 0.0)
            print('0% deteoriation')

        print("Inputs set successfully.")

        # Run the model
        if not gspdll.CalculateSteadyStatePoint(False):
            print("Failed to run the model.")
            return [None]*8
        print("Model run successfully.")

        # Extract outputs
        output_values = []
        for index in range(1, 9):  # Assuming indices 1 to 7 are valid
            dv = ctypes.c_double(0.0)
            pdv = ctypes.pointer(dv)
            if gspdll.GetOutputDataParameterValueByIndex(index, pdv, False):
                output_values.append(pdv.contents.value)
                print(f"Output {index}: {pdv.contents.value}")
            else:
                output_values.append(None)
                print(f"Failed to retrieve output {index}.")
        return output_values
    except Exception as e:
        print(f"Error processing row: {e}")
        return None

