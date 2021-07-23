#include "complementaryfilter.hpp"

std::unique_ptr<CFilter> filt_;

static PyObject *init(PyObject *self, PyObject *args){
    uint16_t x, y, L1, L2, maxL;
    float_t th_pos, th_neg, alp, lam;
    if (!PyArg_ParseTuple(args, "HHffffHHH", &x, &y, &th_pos, &th_neg, &alp, &lam, &L1, &L2, &maxL))
        return NULL;
    filt_ =  std::make_unique<CFilter>(x, y, th_pos, th_neg, alp, lam, L1, L2, maxL);
    return Py_True;
}

static PyObject *filterEv(PyObject *self, PyObject *args){
    PyObject *chunk, *ex_chunk; 
    if (!PyArg_ParseTuple(args, "OO", &chunk, &ex_chunk))
        return NULL;
    PyArrayObject *chunk_array, *ex_chunk_array; 
    chunk_array = reinterpret_cast<PyArrayObject *>(PyArray_FROM_OF(chunk, NPY_ARRAY_C_CONTIGUOUS));
    ex_chunk_array = reinterpret_cast<PyArrayObject *>(PyArray_FROM_OF(ex_chunk, NPY_ARRAY_C_CONTIGUOUS));
    auto shape = PyArray_SHAPE(chunk_array);
    auto type = PyArray_DTYPE(ex_chunk_array);
    std::vector<Event> in(shape[0]);
    std::vector<IntensityEvent> out;
    std::memcpy(in.data(), PyArray_DATA(chunk_array), sizeof(Event) * shape[0]);
    filt_->processEvent(in, out);
    npy_intp size = static_cast<npy_intp>(out.size());
    PyObject *result_array = PyArray_NewFromDescr(&PyArray_Type, type, 1, &size, nullptr, nullptr, 0, nullptr);
    std::memcpy(PyArray_DATA((PyArrayObject*) result_array ), out.data(), sizeof(IntensityEvent) * size);
    auto stream = PyDict_New();
    PyDict_SetItem(stream, PyUnicode_FromString("ev"), result_array);
    return stream;
}

static PyObject *filterIm(PyObject *self, PyObject *args){
    PyObject *chunk; 
    if (!PyArg_ParseTuple(args, "O", &chunk))
        return NULL;
    PyArrayObject *chunk_array; 
    chunk_array = reinterpret_cast<PyArrayObject *>(PyArray_FROM_OF(chunk, NPY_ARRAY_C_CONTIGUOUS));
    auto shape = PyArray_SHAPE(chunk_array);
    auto type = PyArray_DTYPE(chunk_array);
    std::vector<IntensityEvent> in(shape[0]), out;
    out.reserve(shape[0]);
    std::memcpy(in.data(), PyArray_DATA(chunk_array), sizeof(IntensityEvent) * shape[0]);
    filt_->processIntensityEvent(in, out);
    npy_intp size = static_cast<npy_intp>(out.size());
    PyObject *result_array = PyArray_NewFromDescr(&PyArray_Type, type, 1, &size, nullptr, nullptr, 0, nullptr);
    std::memcpy(PyArray_DATA((PyArrayObject*) result_array), out.data(), sizeof(IntensityEvent) * size);
    auto stream = PyDict_New();
    PyDict_SetItem(stream, PyUnicode_FromString("ev"), result_array);
    return stream;
}


// Doc
PyDoc_STRVAR(
    init_doc,
    "Initialize the filter.\n\n"
    "Parameters\n"
    "----------\n"
    "x: uint16_t\n"
    "   Number of rows\n"
    "y: uint16_t\n"
    "   Number of columns\n"
    "th_pos: float_t\n"
    "   Mean positive threshold\n"
    "th_neg: float_t\n"
    "   Mean negative threshold\n"
    "alp: float_t\n"
    "   Alpha\n"
    "lam: float_t\n"
    "   Lambda\n"
    "L1: uint16_t\n"
    "   l1\n"
    "L2: uint16_t\n"
    "   L2\n"
    "maxL: uint16_t\n"
    "   ADC resolution\n" 
    );
PyDoc_STRVAR(
    filterEv_doc,
    "Filter the DVS events. \n\n"
    "Parameters\n"
    "----------\n"
    "in: std::vector<Event> packed in a Numpy structured array\n"
    "   \n"
    "ex: std::vector<IntensityEvent> packed in a Numpy structured array\n"
    "   Empty Intensity buffer to catch the type\n"
    "Returns\n"
    "----------\n"
    "out: std::vector<IntensityEvent> packed in a Numpy structured array itself packed in a dictionnary\n"
    "   CF result\n");
PyDoc_STRVAR(
    filterIm_doc,
    "Filter the image events. See the python example to convert an image into a stream of Intensity events\n\n"
    "Parameters\n"
    "----------\n"
    "in: std::vector<IntensityEvent> packed in a Numpy structured array\n"
    "   Input buffer\n"
    "Returns\n"
    "----------\n"
    "out: std::vector<IntensityEvent> packed in a Numpy structured array itself packed in a dictionnary\n"
    "   CF result\n");
static char module_docstring[] = "C++ Template filter";

//Module specification
static PyMethodDef module_methods[] = {
    {"filterEv", (PyCFunction)filterEv, METH_VARARGS, filterEv_doc},
    {"filterIm", (PyCFunction)filterIm, METH_VARARGS, filterIm_doc},
    {"init", (PyCFunction)init, METH_VARARGS, init_doc},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef def = {
    PyModuleDef_HEAD_INIT,
    "complementaryfilter",
    module_docstring,
    -1,
    module_methods};

//Initialize module
PyMODINIT_FUNC PyInit_complementaryfilter(void){
    PyObject *m = PyModule_Create(&def);
    if (m == NULL)
        return NULL;
    //numpy functionallity
    import_array();
    return m;
}
