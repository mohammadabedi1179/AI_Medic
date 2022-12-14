??
??
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( ?
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
N
Cast	
x"SrcT	
y"DstT"
SrcTtype"
DstTtype"
Truncatebool( 
8
Const
output"dtype"
valuetensor"
dtypetype
?
Conv2D

input"T
filter"T
output"T"
Ttype:	
2"
strides	list(int)"
use_cudnn_on_gpubool(",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 "-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

?
FusedBatchNormV3
x"T

scale"U
offset"U	
mean"U
variance"U
y"T

batch_mean"U
batch_variance"U
reserve_space_1"U
reserve_space_2"U
reserve_space_3"U"
Ttype:
2"
Utype:
2"
epsilonfloat%??8"&
exponential_avg_factorfloat%  ??";
data_formatstringNHWC:
NHWCNCHWNDHWCNCDHW"
is_trainingbool(
.
Identity

input"T
output"T"	
Ttype
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
?
MaxPool

input"T
output"T"
Ttype0:
2	"
ksize	list(int)(0"
strides	list(int)(0",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 ":
data_formatstringNHWC:
NHWCNCHWNCHW_VECT_C
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(?

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype?
E
Relu
features"T
activations"T"
Ttype:
2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
?
Select
	condition

t"T
e"T
output"T"	
Ttype
H
ShardedFilename
basename	
shard

num_shards
filename
9
Softmax
logits"T
softmax"T"
Ttype:
2
?
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ??
@
StaticRegexFullMatch	
input

output
"
patternstring
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
P
	Transpose
x"T
perm"Tperm
y"T"	
Ttype"
Tpermtype0:
2	
?
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 ?"serve*2.9.12v2.9.0-18-gd8ce9f9c3018??
?
Adam/output_4/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:$*%
shared_nameAdam/output_4/bias/v
y
(Adam/output_4/bias/v/Read/ReadVariableOpReadVariableOpAdam/output_4/bias/v*
_output_shapes
:$*
dtype0
?
Adam/output_4/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:($*'
shared_nameAdam/output_4/kernel/v
?
*Adam/output_4/kernel/v/Read/ReadVariableOpReadVariableOpAdam/output_4/kernel/v*
_output_shapes

:($*
dtype0
?
Adam/output_3/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:$*%
shared_nameAdam/output_3/bias/v
y
(Adam/output_3/bias/v/Read/ReadVariableOpReadVariableOpAdam/output_3/bias/v*
_output_shapes
:$*
dtype0
?
Adam/output_3/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:($*'
shared_nameAdam/output_3/kernel/v
?
*Adam/output_3/kernel/v/Read/ReadVariableOpReadVariableOpAdam/output_3/kernel/v*
_output_shapes

:($*
dtype0
?
Adam/output_2/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:$*%
shared_nameAdam/output_2/bias/v
y
(Adam/output_2/bias/v/Read/ReadVariableOpReadVariableOpAdam/output_2/bias/v*
_output_shapes
:$*
dtype0
?
Adam/output_2/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:($*'
shared_nameAdam/output_2/kernel/v
?
*Adam/output_2/kernel/v/Read/ReadVariableOpReadVariableOpAdam/output_2/kernel/v*
_output_shapes

:($*
dtype0
?
Adam/output_1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:$*%
shared_nameAdam/output_1/bias/v
y
(Adam/output_1/bias/v/Read/ReadVariableOpReadVariableOpAdam/output_1/bias/v*
_output_shapes
:$*
dtype0
?
Adam/output_1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:($*'
shared_nameAdam/output_1/kernel/v
?
*Adam/output_1/kernel/v/Read/ReadVariableOpReadVariableOpAdam/output_1/kernel/v*
_output_shapes

:($*
dtype0
?
Adam/output_0/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:$*%
shared_nameAdam/output_0/bias/v
y
(Adam/output_0/bias/v/Read/ReadVariableOpReadVariableOpAdam/output_0/bias/v*
_output_shapes
:$*
dtype0
?
Adam/output_0/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:($*'
shared_nameAdam/output_0/kernel/v
?
*Adam/output_0/kernel/v/Read/ReadVariableOpReadVariableOpAdam/output_0/kernel/v*
_output_shapes

:($*
dtype0
?
$Adam/1st_fully_connected_of_4/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:(*5
shared_name&$Adam/1st_fully_connected_of_4/bias/v
?
8Adam/1st_fully_connected_of_4/bias/v/Read/ReadVariableOpReadVariableOp$Adam/1st_fully_connected_of_4/bias/v*
_output_shapes
:(*
dtype0
?
&Adam/1st_fully_connected_of_4/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?$(*7
shared_name(&Adam/1st_fully_connected_of_4/kernel/v
?
:Adam/1st_fully_connected_of_4/kernel/v/Read/ReadVariableOpReadVariableOp&Adam/1st_fully_connected_of_4/kernel/v*
_output_shapes
:	?$(*
dtype0
?
$Adam/1st_fully_connected_of_3/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:(*5
shared_name&$Adam/1st_fully_connected_of_3/bias/v
?
8Adam/1st_fully_connected_of_3/bias/v/Read/ReadVariableOpReadVariableOp$Adam/1st_fully_connected_of_3/bias/v*
_output_shapes
:(*
dtype0
?
&Adam/1st_fully_connected_of_3/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?$(*7
shared_name(&Adam/1st_fully_connected_of_3/kernel/v
?
:Adam/1st_fully_connected_of_3/kernel/v/Read/ReadVariableOpReadVariableOp&Adam/1st_fully_connected_of_3/kernel/v*
_output_shapes
:	?$(*
dtype0
?
$Adam/1st_fully_connected_of_2/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:(*5
shared_name&$Adam/1st_fully_connected_of_2/bias/v
?
8Adam/1st_fully_connected_of_2/bias/v/Read/ReadVariableOpReadVariableOp$Adam/1st_fully_connected_of_2/bias/v*
_output_shapes
:(*
dtype0
?
&Adam/1st_fully_connected_of_2/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?$(*7
shared_name(&Adam/1st_fully_connected_of_2/kernel/v
?
:Adam/1st_fully_connected_of_2/kernel/v/Read/ReadVariableOpReadVariableOp&Adam/1st_fully_connected_of_2/kernel/v*
_output_shapes
:	?$(*
dtype0
?
$Adam/1st_fully_connected_of_1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:(*5
shared_name&$Adam/1st_fully_connected_of_1/bias/v
?
8Adam/1st_fully_connected_of_1/bias/v/Read/ReadVariableOpReadVariableOp$Adam/1st_fully_connected_of_1/bias/v*
_output_shapes
:(*
dtype0
?
&Adam/1st_fully_connected_of_1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?$(*7
shared_name(&Adam/1st_fully_connected_of_1/kernel/v
?
:Adam/1st_fully_connected_of_1/kernel/v/Read/ReadVariableOpReadVariableOp&Adam/1st_fully_connected_of_1/kernel/v*
_output_shapes
:	?$(*
dtype0
?
$Adam/1st_fully_connected_of_0/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:(*5
shared_name&$Adam/1st_fully_connected_of_0/bias/v
?
8Adam/1st_fully_connected_of_0/bias/v/Read/ReadVariableOpReadVariableOp$Adam/1st_fully_connected_of_0/bias/v*
_output_shapes
:(*
dtype0
?
&Adam/1st_fully_connected_of_0/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?$(*7
shared_name(&Adam/1st_fully_connected_of_0/kernel/v
?
:Adam/1st_fully_connected_of_0/kernel/v/Read/ReadVariableOpReadVariableOp&Adam/1st_fully_connected_of_0/kernel/v*
_output_shapes
:	?$(*
dtype0
?
!Adam/batch_normalization_5/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *2
shared_name#!Adam/batch_normalization_5/beta/v
?
5Adam/batch_normalization_5/beta/v/Read/ReadVariableOpReadVariableOp!Adam/batch_normalization_5/beta/v*
_output_shapes
: *
dtype0
?
"Adam/batch_normalization_5/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *3
shared_name$"Adam/batch_normalization_5/gamma/v
?
6Adam/batch_normalization_5/gamma/v/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_5/gamma/v*
_output_shapes
: *
dtype0
?
Adam/6th_conv/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *%
shared_nameAdam/6th_conv/bias/v
y
(Adam/6th_conv/bias/v/Read/ReadVariableOpReadVariableOpAdam/6th_conv/bias/v*
_output_shapes
: *
dtype0
?
Adam/6th_conv/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:  *'
shared_nameAdam/6th_conv/kernel/v
?
*Adam/6th_conv/kernel/v/Read/ReadVariableOpReadVariableOpAdam/6th_conv/kernel/v*&
_output_shapes
:  *
dtype0
?
!Adam/batch_normalization_4/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *2
shared_name#!Adam/batch_normalization_4/beta/v
?
5Adam/batch_normalization_4/beta/v/Read/ReadVariableOpReadVariableOp!Adam/batch_normalization_4/beta/v*
_output_shapes
: *
dtype0
?
"Adam/batch_normalization_4/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *3
shared_name$"Adam/batch_normalization_4/gamma/v
?
6Adam/batch_normalization_4/gamma/v/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_4/gamma/v*
_output_shapes
: *
dtype0
?
Adam/5th_conv/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *%
shared_nameAdam/5th_conv/bias/v
y
(Adam/5th_conv/bias/v/Read/ReadVariableOpReadVariableOpAdam/5th_conv/bias/v*
_output_shapes
: *
dtype0
?
Adam/5th_conv/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:  *'
shared_nameAdam/5th_conv/kernel/v
?
*Adam/5th_conv/kernel/v/Read/ReadVariableOpReadVariableOpAdam/5th_conv/kernel/v*&
_output_shapes
:  *
dtype0
?
!Adam/batch_normalization_3/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *2
shared_name#!Adam/batch_normalization_3/beta/v
?
5Adam/batch_normalization_3/beta/v/Read/ReadVariableOpReadVariableOp!Adam/batch_normalization_3/beta/v*
_output_shapes
: *
dtype0
?
"Adam/batch_normalization_3/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *3
shared_name$"Adam/batch_normalization_3/gamma/v
?
6Adam/batch_normalization_3/gamma/v/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_3/gamma/v*
_output_shapes
: *
dtype0
?
Adam/4th_conv/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *%
shared_nameAdam/4th_conv/bias/v
y
(Adam/4th_conv/bias/v/Read/ReadVariableOpReadVariableOpAdam/4th_conv/bias/v*
_output_shapes
: *
dtype0
?
Adam/4th_conv/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *'
shared_nameAdam/4th_conv/kernel/v
?
*Adam/4th_conv/kernel/v/Read/ReadVariableOpReadVariableOpAdam/4th_conv/kernel/v*&
_output_shapes
: *
dtype0
?
Adam/output_4/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:$*%
shared_nameAdam/output_4/bias/m
y
(Adam/output_4/bias/m/Read/ReadVariableOpReadVariableOpAdam/output_4/bias/m*
_output_shapes
:$*
dtype0
?
Adam/output_4/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:($*'
shared_nameAdam/output_4/kernel/m
?
*Adam/output_4/kernel/m/Read/ReadVariableOpReadVariableOpAdam/output_4/kernel/m*
_output_shapes

:($*
dtype0
?
Adam/output_3/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:$*%
shared_nameAdam/output_3/bias/m
y
(Adam/output_3/bias/m/Read/ReadVariableOpReadVariableOpAdam/output_3/bias/m*
_output_shapes
:$*
dtype0
?
Adam/output_3/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:($*'
shared_nameAdam/output_3/kernel/m
?
*Adam/output_3/kernel/m/Read/ReadVariableOpReadVariableOpAdam/output_3/kernel/m*
_output_shapes

:($*
dtype0
?
Adam/output_2/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:$*%
shared_nameAdam/output_2/bias/m
y
(Adam/output_2/bias/m/Read/ReadVariableOpReadVariableOpAdam/output_2/bias/m*
_output_shapes
:$*
dtype0
?
Adam/output_2/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:($*'
shared_nameAdam/output_2/kernel/m
?
*Adam/output_2/kernel/m/Read/ReadVariableOpReadVariableOpAdam/output_2/kernel/m*
_output_shapes

:($*
dtype0
?
Adam/output_1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:$*%
shared_nameAdam/output_1/bias/m
y
(Adam/output_1/bias/m/Read/ReadVariableOpReadVariableOpAdam/output_1/bias/m*
_output_shapes
:$*
dtype0
?
Adam/output_1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:($*'
shared_nameAdam/output_1/kernel/m
?
*Adam/output_1/kernel/m/Read/ReadVariableOpReadVariableOpAdam/output_1/kernel/m*
_output_shapes

:($*
dtype0
?
Adam/output_0/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:$*%
shared_nameAdam/output_0/bias/m
y
(Adam/output_0/bias/m/Read/ReadVariableOpReadVariableOpAdam/output_0/bias/m*
_output_shapes
:$*
dtype0
?
Adam/output_0/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:($*'
shared_nameAdam/output_0/kernel/m
?
*Adam/output_0/kernel/m/Read/ReadVariableOpReadVariableOpAdam/output_0/kernel/m*
_output_shapes

:($*
dtype0
?
$Adam/1st_fully_connected_of_4/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:(*5
shared_name&$Adam/1st_fully_connected_of_4/bias/m
?
8Adam/1st_fully_connected_of_4/bias/m/Read/ReadVariableOpReadVariableOp$Adam/1st_fully_connected_of_4/bias/m*
_output_shapes
:(*
dtype0
?
&Adam/1st_fully_connected_of_4/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?$(*7
shared_name(&Adam/1st_fully_connected_of_4/kernel/m
?
:Adam/1st_fully_connected_of_4/kernel/m/Read/ReadVariableOpReadVariableOp&Adam/1st_fully_connected_of_4/kernel/m*
_output_shapes
:	?$(*
dtype0
?
$Adam/1st_fully_connected_of_3/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:(*5
shared_name&$Adam/1st_fully_connected_of_3/bias/m
?
8Adam/1st_fully_connected_of_3/bias/m/Read/ReadVariableOpReadVariableOp$Adam/1st_fully_connected_of_3/bias/m*
_output_shapes
:(*
dtype0
?
&Adam/1st_fully_connected_of_3/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?$(*7
shared_name(&Adam/1st_fully_connected_of_3/kernel/m
?
:Adam/1st_fully_connected_of_3/kernel/m/Read/ReadVariableOpReadVariableOp&Adam/1st_fully_connected_of_3/kernel/m*
_output_shapes
:	?$(*
dtype0
?
$Adam/1st_fully_connected_of_2/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:(*5
shared_name&$Adam/1st_fully_connected_of_2/bias/m
?
8Adam/1st_fully_connected_of_2/bias/m/Read/ReadVariableOpReadVariableOp$Adam/1st_fully_connected_of_2/bias/m*
_output_shapes
:(*
dtype0
?
&Adam/1st_fully_connected_of_2/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?$(*7
shared_name(&Adam/1st_fully_connected_of_2/kernel/m
?
:Adam/1st_fully_connected_of_2/kernel/m/Read/ReadVariableOpReadVariableOp&Adam/1st_fully_connected_of_2/kernel/m*
_output_shapes
:	?$(*
dtype0
?
$Adam/1st_fully_connected_of_1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:(*5
shared_name&$Adam/1st_fully_connected_of_1/bias/m
?
8Adam/1st_fully_connected_of_1/bias/m/Read/ReadVariableOpReadVariableOp$Adam/1st_fully_connected_of_1/bias/m*
_output_shapes
:(*
dtype0
?
&Adam/1st_fully_connected_of_1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?$(*7
shared_name(&Adam/1st_fully_connected_of_1/kernel/m
?
:Adam/1st_fully_connected_of_1/kernel/m/Read/ReadVariableOpReadVariableOp&Adam/1st_fully_connected_of_1/kernel/m*
_output_shapes
:	?$(*
dtype0
?
$Adam/1st_fully_connected_of_0/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:(*5
shared_name&$Adam/1st_fully_connected_of_0/bias/m
?
8Adam/1st_fully_connected_of_0/bias/m/Read/ReadVariableOpReadVariableOp$Adam/1st_fully_connected_of_0/bias/m*
_output_shapes
:(*
dtype0
?
&Adam/1st_fully_connected_of_0/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?$(*7
shared_name(&Adam/1st_fully_connected_of_0/kernel/m
?
:Adam/1st_fully_connected_of_0/kernel/m/Read/ReadVariableOpReadVariableOp&Adam/1st_fully_connected_of_0/kernel/m*
_output_shapes
:	?$(*
dtype0
?
!Adam/batch_normalization_5/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *2
shared_name#!Adam/batch_normalization_5/beta/m
?
5Adam/batch_normalization_5/beta/m/Read/ReadVariableOpReadVariableOp!Adam/batch_normalization_5/beta/m*
_output_shapes
: *
dtype0
?
"Adam/batch_normalization_5/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *3
shared_name$"Adam/batch_normalization_5/gamma/m
?
6Adam/batch_normalization_5/gamma/m/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_5/gamma/m*
_output_shapes
: *
dtype0
?
Adam/6th_conv/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *%
shared_nameAdam/6th_conv/bias/m
y
(Adam/6th_conv/bias/m/Read/ReadVariableOpReadVariableOpAdam/6th_conv/bias/m*
_output_shapes
: *
dtype0
?
Adam/6th_conv/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:  *'
shared_nameAdam/6th_conv/kernel/m
?
*Adam/6th_conv/kernel/m/Read/ReadVariableOpReadVariableOpAdam/6th_conv/kernel/m*&
_output_shapes
:  *
dtype0
?
!Adam/batch_normalization_4/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *2
shared_name#!Adam/batch_normalization_4/beta/m
?
5Adam/batch_normalization_4/beta/m/Read/ReadVariableOpReadVariableOp!Adam/batch_normalization_4/beta/m*
_output_shapes
: *
dtype0
?
"Adam/batch_normalization_4/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *3
shared_name$"Adam/batch_normalization_4/gamma/m
?
6Adam/batch_normalization_4/gamma/m/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_4/gamma/m*
_output_shapes
: *
dtype0
?
Adam/5th_conv/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *%
shared_nameAdam/5th_conv/bias/m
y
(Adam/5th_conv/bias/m/Read/ReadVariableOpReadVariableOpAdam/5th_conv/bias/m*
_output_shapes
: *
dtype0
?
Adam/5th_conv/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:  *'
shared_nameAdam/5th_conv/kernel/m
?
*Adam/5th_conv/kernel/m/Read/ReadVariableOpReadVariableOpAdam/5th_conv/kernel/m*&
_output_shapes
:  *
dtype0
?
!Adam/batch_normalization_3/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *2
shared_name#!Adam/batch_normalization_3/beta/m
?
5Adam/batch_normalization_3/beta/m/Read/ReadVariableOpReadVariableOp!Adam/batch_normalization_3/beta/m*
_output_shapes
: *
dtype0
?
"Adam/batch_normalization_3/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *3
shared_name$"Adam/batch_normalization_3/gamma/m
?
6Adam/batch_normalization_3/gamma/m/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_3/gamma/m*
_output_shapes
: *
dtype0
?
Adam/4th_conv/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *%
shared_nameAdam/4th_conv/bias/m
y
(Adam/4th_conv/bias/m/Read/ReadVariableOpReadVariableOpAdam/4th_conv/bias/m*
_output_shapes
: *
dtype0
?
Adam/4th_conv/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *'
shared_nameAdam/4th_conv/kernel/m
?
*Adam/4th_conv/kernel/m/Read/ReadVariableOpReadVariableOpAdam/4th_conv/kernel/m*&
_output_shapes
: *
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
b
count_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_1
[
count_1/Read/ReadVariableOpReadVariableOpcount_1*
_output_shapes
: *
dtype0
b
total_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_1
[
total_1/Read/ReadVariableOpReadVariableOptotal_1*
_output_shapes
: *
dtype0
x
Adam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/learning_rate
q
&Adam/learning_rate/Read/ReadVariableOpReadVariableOpAdam/learning_rate*
_output_shapes
: *
dtype0
h

Adam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Adam/decay
a
Adam/decay/Read/ReadVariableOpReadVariableOp
Adam/decay*
_output_shapes
: *
dtype0
j
Adam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_2
c
Adam/beta_2/Read/ReadVariableOpReadVariableOpAdam/beta_2*
_output_shapes
: *
dtype0
j
Adam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_1
c
Adam/beta_1/Read/ReadVariableOpReadVariableOpAdam/beta_1*
_output_shapes
: *
dtype0
f
	Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	Adam/iter
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
_output_shapes
: *
dtype0	
r
output_4/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:$*
shared_nameoutput_4/bias
k
!output_4/bias/Read/ReadVariableOpReadVariableOpoutput_4/bias*
_output_shapes
:$*
dtype0
z
output_4/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:($* 
shared_nameoutput_4/kernel
s
#output_4/kernel/Read/ReadVariableOpReadVariableOpoutput_4/kernel*
_output_shapes

:($*
dtype0
r
output_3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:$*
shared_nameoutput_3/bias
k
!output_3/bias/Read/ReadVariableOpReadVariableOpoutput_3/bias*
_output_shapes
:$*
dtype0
z
output_3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:($* 
shared_nameoutput_3/kernel
s
#output_3/kernel/Read/ReadVariableOpReadVariableOpoutput_3/kernel*
_output_shapes

:($*
dtype0
r
output_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:$*
shared_nameoutput_2/bias
k
!output_2/bias/Read/ReadVariableOpReadVariableOpoutput_2/bias*
_output_shapes
:$*
dtype0
z
output_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:($* 
shared_nameoutput_2/kernel
s
#output_2/kernel/Read/ReadVariableOpReadVariableOpoutput_2/kernel*
_output_shapes

:($*
dtype0
r
output_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:$*
shared_nameoutput_1/bias
k
!output_1/bias/Read/ReadVariableOpReadVariableOpoutput_1/bias*
_output_shapes
:$*
dtype0
z
output_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:($* 
shared_nameoutput_1/kernel
s
#output_1/kernel/Read/ReadVariableOpReadVariableOpoutput_1/kernel*
_output_shapes

:($*
dtype0
r
output_0/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:$*
shared_nameoutput_0/bias
k
!output_0/bias/Read/ReadVariableOpReadVariableOpoutput_0/bias*
_output_shapes
:$*
dtype0
z
output_0/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:($* 
shared_nameoutput_0/kernel
s
#output_0/kernel/Read/ReadVariableOpReadVariableOpoutput_0/kernel*
_output_shapes

:($*
dtype0
?
1st_fully_connected_of_4/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:(*.
shared_name1st_fully_connected_of_4/bias
?
11st_fully_connected_of_4/bias/Read/ReadVariableOpReadVariableOp1st_fully_connected_of_4/bias*
_output_shapes
:(*
dtype0
?
1st_fully_connected_of_4/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?$(*0
shared_name!1st_fully_connected_of_4/kernel
?
31st_fully_connected_of_4/kernel/Read/ReadVariableOpReadVariableOp1st_fully_connected_of_4/kernel*
_output_shapes
:	?$(*
dtype0
?
1st_fully_connected_of_3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:(*.
shared_name1st_fully_connected_of_3/bias
?
11st_fully_connected_of_3/bias/Read/ReadVariableOpReadVariableOp1st_fully_connected_of_3/bias*
_output_shapes
:(*
dtype0
?
1st_fully_connected_of_3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?$(*0
shared_name!1st_fully_connected_of_3/kernel
?
31st_fully_connected_of_3/kernel/Read/ReadVariableOpReadVariableOp1st_fully_connected_of_3/kernel*
_output_shapes
:	?$(*
dtype0
?
1st_fully_connected_of_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:(*.
shared_name1st_fully_connected_of_2/bias
?
11st_fully_connected_of_2/bias/Read/ReadVariableOpReadVariableOp1st_fully_connected_of_2/bias*
_output_shapes
:(*
dtype0
?
1st_fully_connected_of_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?$(*0
shared_name!1st_fully_connected_of_2/kernel
?
31st_fully_connected_of_2/kernel/Read/ReadVariableOpReadVariableOp1st_fully_connected_of_2/kernel*
_output_shapes
:	?$(*
dtype0
?
1st_fully_connected_of_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:(*.
shared_name1st_fully_connected_of_1/bias
?
11st_fully_connected_of_1/bias/Read/ReadVariableOpReadVariableOp1st_fully_connected_of_1/bias*
_output_shapes
:(*
dtype0
?
1st_fully_connected_of_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?$(*0
shared_name!1st_fully_connected_of_1/kernel
?
31st_fully_connected_of_1/kernel/Read/ReadVariableOpReadVariableOp1st_fully_connected_of_1/kernel*
_output_shapes
:	?$(*
dtype0
?
1st_fully_connected_of_0/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:(*.
shared_name1st_fully_connected_of_0/bias
?
11st_fully_connected_of_0/bias/Read/ReadVariableOpReadVariableOp1st_fully_connected_of_0/bias*
_output_shapes
:(*
dtype0
?
1st_fully_connected_of_0/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?$(*0
shared_name!1st_fully_connected_of_0/kernel
?
31st_fully_connected_of_0/kernel/Read/ReadVariableOpReadVariableOp1st_fully_connected_of_0/kernel*
_output_shapes
:	?$(*
dtype0
?
%batch_normalization_5/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape: *6
shared_name'%batch_normalization_5/moving_variance
?
9batch_normalization_5/moving_variance/Read/ReadVariableOpReadVariableOp%batch_normalization_5/moving_variance*
_output_shapes
: *
dtype0
?
!batch_normalization_5/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape: *2
shared_name#!batch_normalization_5/moving_mean
?
5batch_normalization_5/moving_mean/Read/ReadVariableOpReadVariableOp!batch_normalization_5/moving_mean*
_output_shapes
: *
dtype0
?
batch_normalization_5/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *+
shared_namebatch_normalization_5/beta
?
.batch_normalization_5/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_5/beta*
_output_shapes
: *
dtype0
?
batch_normalization_5/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *,
shared_namebatch_normalization_5/gamma
?
/batch_normalization_5/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_5/gamma*
_output_shapes
: *
dtype0
r
6th_conv/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name6th_conv/bias
k
!6th_conv/bias/Read/ReadVariableOpReadVariableOp6th_conv/bias*
_output_shapes
: *
dtype0
?
6th_conv/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:  * 
shared_name6th_conv/kernel
{
#6th_conv/kernel/Read/ReadVariableOpReadVariableOp6th_conv/kernel*&
_output_shapes
:  *
dtype0
?
%batch_normalization_4/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape: *6
shared_name'%batch_normalization_4/moving_variance
?
9batch_normalization_4/moving_variance/Read/ReadVariableOpReadVariableOp%batch_normalization_4/moving_variance*
_output_shapes
: *
dtype0
?
!batch_normalization_4/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape: *2
shared_name#!batch_normalization_4/moving_mean
?
5batch_normalization_4/moving_mean/Read/ReadVariableOpReadVariableOp!batch_normalization_4/moving_mean*
_output_shapes
: *
dtype0
?
batch_normalization_4/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *+
shared_namebatch_normalization_4/beta
?
.batch_normalization_4/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_4/beta*
_output_shapes
: *
dtype0
?
batch_normalization_4/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *,
shared_namebatch_normalization_4/gamma
?
/batch_normalization_4/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_4/gamma*
_output_shapes
: *
dtype0
r
5th_conv/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name5th_conv/bias
k
!5th_conv/bias/Read/ReadVariableOpReadVariableOp5th_conv/bias*
_output_shapes
: *
dtype0
?
5th_conv/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:  * 
shared_name5th_conv/kernel
{
#5th_conv/kernel/Read/ReadVariableOpReadVariableOp5th_conv/kernel*&
_output_shapes
:  *
dtype0
?
%batch_normalization_3/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape: *6
shared_name'%batch_normalization_3/moving_variance
?
9batch_normalization_3/moving_variance/Read/ReadVariableOpReadVariableOp%batch_normalization_3/moving_variance*
_output_shapes
: *
dtype0
?
!batch_normalization_3/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape: *2
shared_name#!batch_normalization_3/moving_mean
?
5batch_normalization_3/moving_mean/Read/ReadVariableOpReadVariableOp!batch_normalization_3/moving_mean*
_output_shapes
: *
dtype0
?
batch_normalization_3/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *+
shared_namebatch_normalization_3/beta
?
.batch_normalization_3/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_3/beta*
_output_shapes
: *
dtype0
?
batch_normalization_3/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *,
shared_namebatch_normalization_3/gamma
?
/batch_normalization_3/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_3/gamma*
_output_shapes
: *
dtype0
r
4th_conv/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name4th_conv/bias
k
!4th_conv/bias/Read/ReadVariableOpReadVariableOp4th_conv/bias*
_output_shapes
: *
dtype0
?
4th_conv/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: * 
shared_name4th_conv/kernel
{
#4th_conv/kernel/Read/ReadVariableOpReadVariableOp4th_conv/kernel*&
_output_shapes
: *
dtype0

NoOpNoOp
??
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*??
value??B?? B??
?
layer-0
layer_with_weights-0
layer-1
layer-2
layer_with_weights-1
layer-3
layer_with_weights-2
layer-4
layer-5
layer_with_weights-3
layer-6
layer_with_weights-4
layer-7
	layer-8

layer_with_weights-5

layer-9
layer-10
layer_with_weights-6
layer-11
layer_with_weights-7
layer-12
layer_with_weights-8
layer-13
layer_with_weights-9
layer-14
layer_with_weights-10
layer-15
layer-16
layer-17
layer-18
layer-19
layer-20
layer_with_weights-11
layer-21
layer_with_weights-12
layer-22
layer_with_weights-13
layer-23
layer_with_weights-14
layer-24
layer_with_weights-15
layer-25
layer-26
layer-27
	variables
trainable_variables
regularization_losses
 	keras_api
!__call__
*"&call_and_return_all_conditional_losses
#_default_save_signature
$	optimizer
%
signatures*
* 
?
&	variables
'trainable_variables
(regularization_losses
)	keras_api
*__call__
*+&call_and_return_all_conditional_losses

,kernel
-bias
 ._jit_compiled_convolution_op*
?
/	variables
0trainable_variables
1regularization_losses
2	keras_api
3__call__
*4&call_and_return_all_conditional_losses* 
?
5	variables
6trainable_variables
7regularization_losses
8	keras_api
9__call__
*:&call_and_return_all_conditional_losses
;axis
	<gamma
=beta
>moving_mean
?moving_variance*
?
@	variables
Atrainable_variables
Bregularization_losses
C	keras_api
D__call__
*E&call_and_return_all_conditional_losses

Fkernel
Gbias
 H_jit_compiled_convolution_op*
?
I	variables
Jtrainable_variables
Kregularization_losses
L	keras_api
M__call__
*N&call_and_return_all_conditional_losses* 
?
O	variables
Ptrainable_variables
Qregularization_losses
R	keras_api
S__call__
*T&call_and_return_all_conditional_losses
Uaxis
	Vgamma
Wbeta
Xmoving_mean
Ymoving_variance*
?
Z	variables
[trainable_variables
\regularization_losses
]	keras_api
^__call__
*_&call_and_return_all_conditional_losses

`kernel
abias
 b_jit_compiled_convolution_op*
?
c	variables
dtrainable_variables
eregularization_losses
f	keras_api
g__call__
*h&call_and_return_all_conditional_losses* 
?
i	variables
jtrainable_variables
kregularization_losses
l	keras_api
m__call__
*n&call_and_return_all_conditional_losses
oaxis
	pgamma
qbeta
rmoving_mean
smoving_variance*
?
t	variables
utrainable_variables
vregularization_losses
w	keras_api
x__call__
*y&call_and_return_all_conditional_losses* 
?
z	variables
{trainable_variables
|regularization_losses
}	keras_api
~__call__
*&call_and_return_all_conditional_losses
?kernel
	?bias*
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses
?kernel
	?bias*
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses
?kernel
	?bias*
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses
?kernel
	?bias*
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses
?kernel
	?bias*
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses
?_random_generator* 
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses
?_random_generator* 
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses
?_random_generator* 
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses
?_random_generator* 
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses
?_random_generator* 
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses
?kernel
	?bias*
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses
?kernel
	?bias*
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses
?kernel
	?bias*
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses
?kernel
	?bias*
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses
?kernel
	?bias*

?	keras_api* 

?	keras_api* 
?
,0
-1
<2
=3
>4
?5
F6
G7
V8
W9
X10
Y11
`12
a13
p14
q15
r16
s17
?18
?19
?20
?21
?22
?23
?24
?25
?26
?27
?28
?29
?30
?31
?32
?33
?34
?35
?36
?37*
?
,0
-1
<2
=3
F4
G5
V6
W7
`8
a9
p10
q11
?12
?13
?14
?15
?16
?17
?18
?19
?20
?21
?22
?23
?24
?25
?26
?27
?28
?29
?30
?31*
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
	variables
trainable_variables
regularization_losses
!__call__
#_default_save_signature
*"&call_and_return_all_conditional_losses
&""call_and_return_conditional_losses*
:
?trace_0
?trace_1
?trace_2
?trace_3* 
:
?trace_0
?trace_1
?trace_2
?trace_3* 
* 
?
	?iter
?beta_1
?beta_2

?decay
?learning_rate,m?-m?<m?=m?Fm?Gm?Vm?Wm?`m?am?pm?qm?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?,v?-v?<v?=v?Fv?Gv?Vv?Wv?`v?av?pv?qv?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?*

?serving_default* 

,0
-1*

,0
-1*
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
&	variables
'trainable_variables
(regularization_losses
*__call__
*+&call_and_return_all_conditional_losses
&+"call_and_return_conditional_losses*

?trace_0* 

?trace_0* 
_Y
VARIABLE_VALUE4th_conv/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUE4th_conv/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
/	variables
0trainable_variables
1regularization_losses
3__call__
*4&call_and_return_all_conditional_losses
&4"call_and_return_conditional_losses* 

?trace_0* 

?trace_0* 
 
<0
=1
>2
?3*

<0
=1*
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
5	variables
6trainable_variables
7regularization_losses
9__call__
*:&call_and_return_all_conditional_losses
&:"call_and_return_conditional_losses*

?trace_0
?trace_1* 

?trace_0
?trace_1* 
* 
jd
VARIABLE_VALUEbatch_normalization_3/gamma5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUE*
hb
VARIABLE_VALUEbatch_normalization_3/beta4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUE*
vp
VARIABLE_VALUE!batch_normalization_3/moving_mean;layer_with_weights-1/moving_mean/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUE%batch_normalization_3/moving_variance?layer_with_weights-1/moving_variance/.ATTRIBUTES/VARIABLE_VALUE*

F0
G1*

F0
G1*
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
@	variables
Atrainable_variables
Bregularization_losses
D__call__
*E&call_and_return_all_conditional_losses
&E"call_and_return_conditional_losses*

?trace_0* 

?trace_0* 
_Y
VARIABLE_VALUE5th_conv/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUE5th_conv/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
I	variables
Jtrainable_variables
Kregularization_losses
M__call__
*N&call_and_return_all_conditional_losses
&N"call_and_return_conditional_losses* 

?trace_0* 

?trace_0* 
 
V0
W1
X2
Y3*

V0
W1*
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
O	variables
Ptrainable_variables
Qregularization_losses
S__call__
*T&call_and_return_all_conditional_losses
&T"call_and_return_conditional_losses*

?trace_0
?trace_1* 

?trace_0
?trace_1* 
* 
jd
VARIABLE_VALUEbatch_normalization_4/gamma5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUE*
hb
VARIABLE_VALUEbatch_normalization_4/beta4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUE*
vp
VARIABLE_VALUE!batch_normalization_4/moving_mean;layer_with_weights-3/moving_mean/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUE%batch_normalization_4/moving_variance?layer_with_weights-3/moving_variance/.ATTRIBUTES/VARIABLE_VALUE*

`0
a1*

`0
a1*
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
Z	variables
[trainable_variables
\regularization_losses
^__call__
*_&call_and_return_all_conditional_losses
&_"call_and_return_conditional_losses*

?trace_0* 

?trace_0* 
_Y
VARIABLE_VALUE6th_conv/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUE6th_conv/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
c	variables
dtrainable_variables
eregularization_losses
g__call__
*h&call_and_return_all_conditional_losses
&h"call_and_return_conditional_losses* 

?trace_0* 

?trace_0* 
 
p0
q1
r2
s3*

p0
q1*
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
i	variables
jtrainable_variables
kregularization_losses
m__call__
*n&call_and_return_all_conditional_losses
&n"call_and_return_conditional_losses*

?trace_0
?trace_1* 

?trace_0
?trace_1* 
* 
jd
VARIABLE_VALUEbatch_normalization_5/gamma5layer_with_weights-5/gamma/.ATTRIBUTES/VARIABLE_VALUE*
hb
VARIABLE_VALUEbatch_normalization_5/beta4layer_with_weights-5/beta/.ATTRIBUTES/VARIABLE_VALUE*
vp
VARIABLE_VALUE!batch_normalization_5/moving_mean;layer_with_weights-5/moving_mean/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUE%batch_normalization_5/moving_variance?layer_with_weights-5/moving_variance/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
t	variables
utrainable_variables
vregularization_losses
x__call__
*y&call_and_return_all_conditional_losses
&y"call_and_return_conditional_losses* 

?trace_0* 

?trace_0* 

?0
?1*

?0
?1*
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
z	variables
{trainable_variables
|regularization_losses
~__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*

?trace_0* 

?trace_0* 
oi
VARIABLE_VALUE1st_fully_connected_of_0/kernel6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUE*
ke
VARIABLE_VALUE1st_fully_connected_of_0/bias4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUE*

?0
?1*

?0
?1*
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses*

?trace_0* 

?trace_0* 
oi
VARIABLE_VALUE1st_fully_connected_of_1/kernel6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUE*
ke
VARIABLE_VALUE1st_fully_connected_of_1/bias4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUE*

?0
?1*

?0
?1*
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses*

?trace_0* 

?trace_0* 
oi
VARIABLE_VALUE1st_fully_connected_of_2/kernel6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUE*
ke
VARIABLE_VALUE1st_fully_connected_of_2/bias4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUE*

?0
?1*

?0
?1*
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses*

?trace_0* 

?trace_0* 
oi
VARIABLE_VALUE1st_fully_connected_of_3/kernel6layer_with_weights-9/kernel/.ATTRIBUTES/VARIABLE_VALUE*
ke
VARIABLE_VALUE1st_fully_connected_of_3/bias4layer_with_weights-9/bias/.ATTRIBUTES/VARIABLE_VALUE*

?0
?1*

?0
?1*
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses*

?trace_0* 

?trace_0* 
pj
VARIABLE_VALUE1st_fully_connected_of_4/kernel7layer_with_weights-10/kernel/.ATTRIBUTES/VARIABLE_VALUE*
lf
VARIABLE_VALUE1st_fully_connected_of_4/bias5layer_with_weights-10/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses* 

?trace_0
?trace_1* 

?trace_0
?trace_1* 
* 
* 
* 
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses* 

?trace_0
?trace_1* 

?trace_0
?trace_1* 
* 
* 
* 
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses* 

?trace_0
?trace_1* 

?trace_0
?trace_1* 
* 
* 
* 
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses* 

?trace_0
?trace_1* 

?trace_0
?trace_1* 
* 
* 
* 
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses* 

?trace_0
?trace_1* 

?trace_0
?trace_1* 
* 

?0
?1*

?0
?1*
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses*

?trace_0* 

?trace_0* 
`Z
VARIABLE_VALUEoutput_0/kernel7layer_with_weights-11/kernel/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUEoutput_0/bias5layer_with_weights-11/bias/.ATTRIBUTES/VARIABLE_VALUE*

?0
?1*

?0
?1*
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses*

?trace_0* 

?trace_0* 
`Z
VARIABLE_VALUEoutput_1/kernel7layer_with_weights-12/kernel/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUEoutput_1/bias5layer_with_weights-12/bias/.ATTRIBUTES/VARIABLE_VALUE*

?0
?1*

?0
?1*
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses*

?trace_0* 

?trace_0* 
`Z
VARIABLE_VALUEoutput_2/kernel7layer_with_weights-13/kernel/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUEoutput_2/bias5layer_with_weights-13/bias/.ATTRIBUTES/VARIABLE_VALUE*

?0
?1*

?0
?1*
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses*

?trace_0* 

?trace_0* 
`Z
VARIABLE_VALUEoutput_3/kernel7layer_with_weights-14/kernel/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUEoutput_3/bias5layer_with_weights-14/bias/.ATTRIBUTES/VARIABLE_VALUE*

?0
?1*

?0
?1*
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses*

?trace_0* 

?trace_0* 
`Z
VARIABLE_VALUEoutput_4/kernel7layer_with_weights-15/kernel/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUEoutput_4/bias5layer_with_weights-15/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
.
>0
?1
X2
Y3
r4
s5*
?
0
1
2
3
4
5
6
7
	8

9
10
11
12
13
14
15
16
17
18
19
20
21
22
23
24
25
26
27*

?0
?1*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
LF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE*
PJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE*
PJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE*
NH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE*
^X
VARIABLE_VALUEAdam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

>0
?1*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

X0
Y1*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

r0
s1*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
<
?	variables
?	keras_api

?total

?count*
M
?	variables
?	keras_api

?total

?count
?
_fn_kwargs*

?0
?1*

?	variables*
UO
VARIABLE_VALUEtotal_14keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEcount_14keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE*

?0
?1*

?	variables*
SM
VARIABLE_VALUEtotal4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEcount4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE*
* 
?|
VARIABLE_VALUEAdam/4th_conv/kernel/mRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUEAdam/4th_conv/bias/mPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE"Adam/batch_normalization_3/gamma/mQlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE!Adam/batch_normalization_3/beta/mPlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
?|
VARIABLE_VALUEAdam/5th_conv/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUEAdam/5th_conv/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE"Adam/batch_normalization_4/gamma/mQlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE!Adam/batch_normalization_4/beta/mPlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
?|
VARIABLE_VALUEAdam/6th_conv/kernel/mRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUEAdam/6th_conv/bias/mPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE"Adam/batch_normalization_5/gamma/mQlayer_with_weights-5/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE!Adam/batch_normalization_5/beta/mPlayer_with_weights-5/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE&Adam/1st_fully_connected_of_0/kernel/mRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE$Adam/1st_fully_connected_of_0/bias/mPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE&Adam/1st_fully_connected_of_1/kernel/mRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE$Adam/1st_fully_connected_of_1/bias/mPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE&Adam/1st_fully_connected_of_2/kernel/mRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE$Adam/1st_fully_connected_of_2/bias/mPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE&Adam/1st_fully_connected_of_3/kernel/mRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE$Adam/1st_fully_connected_of_3/bias/mPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE&Adam/1st_fully_connected_of_4/kernel/mSlayer_with_weights-10/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE$Adam/1st_fully_connected_of_4/bias/mQlayer_with_weights-10/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
?}
VARIABLE_VALUEAdam/output_0/kernel/mSlayer_with_weights-11/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUEAdam/output_0/bias/mQlayer_with_weights-11/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
?}
VARIABLE_VALUEAdam/output_1/kernel/mSlayer_with_weights-12/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUEAdam/output_1/bias/mQlayer_with_weights-12/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
?}
VARIABLE_VALUEAdam/output_2/kernel/mSlayer_with_weights-13/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUEAdam/output_2/bias/mQlayer_with_weights-13/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
?}
VARIABLE_VALUEAdam/output_3/kernel/mSlayer_with_weights-14/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUEAdam/output_3/bias/mQlayer_with_weights-14/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
?}
VARIABLE_VALUEAdam/output_4/kernel/mSlayer_with_weights-15/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUEAdam/output_4/bias/mQlayer_with_weights-15/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
?|
VARIABLE_VALUEAdam/4th_conv/kernel/vRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUEAdam/4th_conv/bias/vPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE"Adam/batch_normalization_3/gamma/vQlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE!Adam/batch_normalization_3/beta/vPlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
?|
VARIABLE_VALUEAdam/5th_conv/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUEAdam/5th_conv/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE"Adam/batch_normalization_4/gamma/vQlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE!Adam/batch_normalization_4/beta/vPlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
?|
VARIABLE_VALUEAdam/6th_conv/kernel/vRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUEAdam/6th_conv/bias/vPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE"Adam/batch_normalization_5/gamma/vQlayer_with_weights-5/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE!Adam/batch_normalization_5/beta/vPlayer_with_weights-5/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE&Adam/1st_fully_connected_of_0/kernel/vRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE$Adam/1st_fully_connected_of_0/bias/vPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE&Adam/1st_fully_connected_of_1/kernel/vRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE$Adam/1st_fully_connected_of_1/bias/vPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE&Adam/1st_fully_connected_of_2/kernel/vRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE$Adam/1st_fully_connected_of_2/bias/vPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE&Adam/1st_fully_connected_of_3/kernel/vRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE$Adam/1st_fully_connected_of_3/bias/vPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE&Adam/1st_fully_connected_of_4/kernel/vSlayer_with_weights-10/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE$Adam/1st_fully_connected_of_4/bias/vQlayer_with_weights-10/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
?}
VARIABLE_VALUEAdam/output_0/kernel/vSlayer_with_weights-11/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUEAdam/output_0/bias/vQlayer_with_weights-11/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
?}
VARIABLE_VALUEAdam/output_1/kernel/vSlayer_with_weights-12/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUEAdam/output_1/bias/vQlayer_with_weights-12/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
?}
VARIABLE_VALUEAdam/output_2/kernel/vSlayer_with_weights-13/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUEAdam/output_2/bias/vQlayer_with_weights-13/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
?}
VARIABLE_VALUEAdam/output_3/kernel/vSlayer_with_weights-14/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUEAdam/output_3/bias/vQlayer_with_weights-14/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
?}
VARIABLE_VALUEAdam/output_4/kernel/vSlayer_with_weights-15/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUEAdam/output_4/bias/vQlayer_with_weights-15/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
?
serving_default_imagesPlaceholder*/
_output_shapes
:?????????dd*
dtype0*$
shape:?????????dd
?

StatefulPartitionedCallStatefulPartitionedCallserving_default_images4th_conv/kernel4th_conv/biasbatch_normalization_3/gammabatch_normalization_3/beta!batch_normalization_3/moving_mean%batch_normalization_3/moving_variance5th_conv/kernel5th_conv/biasbatch_normalization_4/gammabatch_normalization_4/beta!batch_normalization_4/moving_mean%batch_normalization_4/moving_variance6th_conv/kernel6th_conv/biasbatch_normalization_5/gammabatch_normalization_5/beta!batch_normalization_5/moving_mean%batch_normalization_5/moving_variance1st_fully_connected_of_4/kernel1st_fully_connected_of_4/bias1st_fully_connected_of_3/kernel1st_fully_connected_of_3/bias1st_fully_connected_of_2/kernel1st_fully_connected_of_2/bias1st_fully_connected_of_1/kernel1st_fully_connected_of_1/bias1st_fully_connected_of_0/kernel1st_fully_connected_of_0/biasoutput_0/kerneloutput_0/biasoutput_1/kerneloutput_1/biasoutput_2/kerneloutput_2/biasoutput_3/kerneloutput_3/biasoutput_4/kerneloutput_4/bias*2
Tin+
)2'*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????$*H
_read_only_resource_inputs*
(&	
 !"#$%&*-
config_proto

CPU

GPU 2J 8? *-
f(R&
$__inference_signature_wrapper_107345
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?+
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename#4th_conv/kernel/Read/ReadVariableOp!4th_conv/bias/Read/ReadVariableOp/batch_normalization_3/gamma/Read/ReadVariableOp.batch_normalization_3/beta/Read/ReadVariableOp5batch_normalization_3/moving_mean/Read/ReadVariableOp9batch_normalization_3/moving_variance/Read/ReadVariableOp#5th_conv/kernel/Read/ReadVariableOp!5th_conv/bias/Read/ReadVariableOp/batch_normalization_4/gamma/Read/ReadVariableOp.batch_normalization_4/beta/Read/ReadVariableOp5batch_normalization_4/moving_mean/Read/ReadVariableOp9batch_normalization_4/moving_variance/Read/ReadVariableOp#6th_conv/kernel/Read/ReadVariableOp!6th_conv/bias/Read/ReadVariableOp/batch_normalization_5/gamma/Read/ReadVariableOp.batch_normalization_5/beta/Read/ReadVariableOp5batch_normalization_5/moving_mean/Read/ReadVariableOp9batch_normalization_5/moving_variance/Read/ReadVariableOp31st_fully_connected_of_0/kernel/Read/ReadVariableOp11st_fully_connected_of_0/bias/Read/ReadVariableOp31st_fully_connected_of_1/kernel/Read/ReadVariableOp11st_fully_connected_of_1/bias/Read/ReadVariableOp31st_fully_connected_of_2/kernel/Read/ReadVariableOp11st_fully_connected_of_2/bias/Read/ReadVariableOp31st_fully_connected_of_3/kernel/Read/ReadVariableOp11st_fully_connected_of_3/bias/Read/ReadVariableOp31st_fully_connected_of_4/kernel/Read/ReadVariableOp11st_fully_connected_of_4/bias/Read/ReadVariableOp#output_0/kernel/Read/ReadVariableOp!output_0/bias/Read/ReadVariableOp#output_1/kernel/Read/ReadVariableOp!output_1/bias/Read/ReadVariableOp#output_2/kernel/Read/ReadVariableOp!output_2/bias/Read/ReadVariableOp#output_3/kernel/Read/ReadVariableOp!output_3/bias/Read/ReadVariableOp#output_4/kernel/Read/ReadVariableOp!output_4/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOp*Adam/4th_conv/kernel/m/Read/ReadVariableOp(Adam/4th_conv/bias/m/Read/ReadVariableOp6Adam/batch_normalization_3/gamma/m/Read/ReadVariableOp5Adam/batch_normalization_3/beta/m/Read/ReadVariableOp*Adam/5th_conv/kernel/m/Read/ReadVariableOp(Adam/5th_conv/bias/m/Read/ReadVariableOp6Adam/batch_normalization_4/gamma/m/Read/ReadVariableOp5Adam/batch_normalization_4/beta/m/Read/ReadVariableOp*Adam/6th_conv/kernel/m/Read/ReadVariableOp(Adam/6th_conv/bias/m/Read/ReadVariableOp6Adam/batch_normalization_5/gamma/m/Read/ReadVariableOp5Adam/batch_normalization_5/beta/m/Read/ReadVariableOp:Adam/1st_fully_connected_of_0/kernel/m/Read/ReadVariableOp8Adam/1st_fully_connected_of_0/bias/m/Read/ReadVariableOp:Adam/1st_fully_connected_of_1/kernel/m/Read/ReadVariableOp8Adam/1st_fully_connected_of_1/bias/m/Read/ReadVariableOp:Adam/1st_fully_connected_of_2/kernel/m/Read/ReadVariableOp8Adam/1st_fully_connected_of_2/bias/m/Read/ReadVariableOp:Adam/1st_fully_connected_of_3/kernel/m/Read/ReadVariableOp8Adam/1st_fully_connected_of_3/bias/m/Read/ReadVariableOp:Adam/1st_fully_connected_of_4/kernel/m/Read/ReadVariableOp8Adam/1st_fully_connected_of_4/bias/m/Read/ReadVariableOp*Adam/output_0/kernel/m/Read/ReadVariableOp(Adam/output_0/bias/m/Read/ReadVariableOp*Adam/output_1/kernel/m/Read/ReadVariableOp(Adam/output_1/bias/m/Read/ReadVariableOp*Adam/output_2/kernel/m/Read/ReadVariableOp(Adam/output_2/bias/m/Read/ReadVariableOp*Adam/output_3/kernel/m/Read/ReadVariableOp(Adam/output_3/bias/m/Read/ReadVariableOp*Adam/output_4/kernel/m/Read/ReadVariableOp(Adam/output_4/bias/m/Read/ReadVariableOp*Adam/4th_conv/kernel/v/Read/ReadVariableOp(Adam/4th_conv/bias/v/Read/ReadVariableOp6Adam/batch_normalization_3/gamma/v/Read/ReadVariableOp5Adam/batch_normalization_3/beta/v/Read/ReadVariableOp*Adam/5th_conv/kernel/v/Read/ReadVariableOp(Adam/5th_conv/bias/v/Read/ReadVariableOp6Adam/batch_normalization_4/gamma/v/Read/ReadVariableOp5Adam/batch_normalization_4/beta/v/Read/ReadVariableOp*Adam/6th_conv/kernel/v/Read/ReadVariableOp(Adam/6th_conv/bias/v/Read/ReadVariableOp6Adam/batch_normalization_5/gamma/v/Read/ReadVariableOp5Adam/batch_normalization_5/beta/v/Read/ReadVariableOp:Adam/1st_fully_connected_of_0/kernel/v/Read/ReadVariableOp8Adam/1st_fully_connected_of_0/bias/v/Read/ReadVariableOp:Adam/1st_fully_connected_of_1/kernel/v/Read/ReadVariableOp8Adam/1st_fully_connected_of_1/bias/v/Read/ReadVariableOp:Adam/1st_fully_connected_of_2/kernel/v/Read/ReadVariableOp8Adam/1st_fully_connected_of_2/bias/v/Read/ReadVariableOp:Adam/1st_fully_connected_of_3/kernel/v/Read/ReadVariableOp8Adam/1st_fully_connected_of_3/bias/v/Read/ReadVariableOp:Adam/1st_fully_connected_of_4/kernel/v/Read/ReadVariableOp8Adam/1st_fully_connected_of_4/bias/v/Read/ReadVariableOp*Adam/output_0/kernel/v/Read/ReadVariableOp(Adam/output_0/bias/v/Read/ReadVariableOp*Adam/output_1/kernel/v/Read/ReadVariableOp(Adam/output_1/bias/v/Read/ReadVariableOp*Adam/output_2/kernel/v/Read/ReadVariableOp(Adam/output_2/bias/v/Read/ReadVariableOp*Adam/output_3/kernel/v/Read/ReadVariableOp(Adam/output_3/bias/v/Read/ReadVariableOp*Adam/output_4/kernel/v/Read/ReadVariableOp(Adam/output_4/bias/v/Read/ReadVariableOpConst*|
Tinu
s2q	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *(
f#R!
__inference__traced_save_108830
?
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filename4th_conv/kernel4th_conv/biasbatch_normalization_3/gammabatch_normalization_3/beta!batch_normalization_3/moving_mean%batch_normalization_3/moving_variance5th_conv/kernel5th_conv/biasbatch_normalization_4/gammabatch_normalization_4/beta!batch_normalization_4/moving_mean%batch_normalization_4/moving_variance6th_conv/kernel6th_conv/biasbatch_normalization_5/gammabatch_normalization_5/beta!batch_normalization_5/moving_mean%batch_normalization_5/moving_variance1st_fully_connected_of_0/kernel1st_fully_connected_of_0/bias1st_fully_connected_of_1/kernel1st_fully_connected_of_1/bias1st_fully_connected_of_2/kernel1st_fully_connected_of_2/bias1st_fully_connected_of_3/kernel1st_fully_connected_of_3/bias1st_fully_connected_of_4/kernel1st_fully_connected_of_4/biasoutput_0/kerneloutput_0/biasoutput_1/kerneloutput_1/biasoutput_2/kerneloutput_2/biasoutput_3/kerneloutput_3/biasoutput_4/kerneloutput_4/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratetotal_1count_1totalcountAdam/4th_conv/kernel/mAdam/4th_conv/bias/m"Adam/batch_normalization_3/gamma/m!Adam/batch_normalization_3/beta/mAdam/5th_conv/kernel/mAdam/5th_conv/bias/m"Adam/batch_normalization_4/gamma/m!Adam/batch_normalization_4/beta/mAdam/6th_conv/kernel/mAdam/6th_conv/bias/m"Adam/batch_normalization_5/gamma/m!Adam/batch_normalization_5/beta/m&Adam/1st_fully_connected_of_0/kernel/m$Adam/1st_fully_connected_of_0/bias/m&Adam/1st_fully_connected_of_1/kernel/m$Adam/1st_fully_connected_of_1/bias/m&Adam/1st_fully_connected_of_2/kernel/m$Adam/1st_fully_connected_of_2/bias/m&Adam/1st_fully_connected_of_3/kernel/m$Adam/1st_fully_connected_of_3/bias/m&Adam/1st_fully_connected_of_4/kernel/m$Adam/1st_fully_connected_of_4/bias/mAdam/output_0/kernel/mAdam/output_0/bias/mAdam/output_1/kernel/mAdam/output_1/bias/mAdam/output_2/kernel/mAdam/output_2/bias/mAdam/output_3/kernel/mAdam/output_3/bias/mAdam/output_4/kernel/mAdam/output_4/bias/mAdam/4th_conv/kernel/vAdam/4th_conv/bias/v"Adam/batch_normalization_3/gamma/v!Adam/batch_normalization_3/beta/vAdam/5th_conv/kernel/vAdam/5th_conv/bias/v"Adam/batch_normalization_4/gamma/v!Adam/batch_normalization_4/beta/vAdam/6th_conv/kernel/vAdam/6th_conv/bias/v"Adam/batch_normalization_5/gamma/v!Adam/batch_normalization_5/beta/v&Adam/1st_fully_connected_of_0/kernel/v$Adam/1st_fully_connected_of_0/bias/v&Adam/1st_fully_connected_of_1/kernel/v$Adam/1st_fully_connected_of_1/bias/v&Adam/1st_fully_connected_of_2/kernel/v$Adam/1st_fully_connected_of_2/bias/v&Adam/1st_fully_connected_of_3/kernel/v$Adam/1st_fully_connected_of_3/bias/v&Adam/1st_fully_connected_of_4/kernel/v$Adam/1st_fully_connected_of_4/bias/vAdam/output_0/kernel/vAdam/output_0/bias/vAdam/output_1/kernel/vAdam/output_1/bias/vAdam/output_2/kernel/vAdam/output_2/bias/vAdam/output_3/kernel/vAdam/output_3/bias/vAdam/output_4/kernel/vAdam/output_4/bias/v*{
Tint
r2p*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *+
f&R$
"__inference__traced_restore_109173??
?
?
D__inference_4th_conv_layer_call_and_return_conditional_losses_106052

inputs8
conv2d_readvariableop_resource: -
biasadd_readvariableop_resource: 
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype0?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????dd *
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????dd X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:?????????dd i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:?????????dd w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????dd: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????dd
 
_user_specified_nameinputs
?
\
@__inference_flat_layer_call_and_return_conditional_losses_106128

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"????   ]
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:??????????$Y
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:??????????$"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:????????? :W S
/
_output_shapes
:????????? 
 
_user_specified_nameinputs
?	
b
C__inference_dropout_layer_call_and_return_conditional_losses_106493

inputs
identity?R
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??d
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:?????????(C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:?????????(*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????(o
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????(i
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:?????????(Y
IdentityIdentitydropout/Mul_1:z:0*
T0*'
_output_shapes
:?????????("
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:?????????(:O K
'
_output_shapes
:?????????(
 
_user_specified_nameinputs
?
`
D__inference_4th_pool_layer_call_and_return_conditional_losses_107882

inputs
identity?
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4????????????????????????????????????*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
?
)__inference_5th_conv_layer_call_fn_107953

inputs!
unknown:  
	unknown_0: 
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????22 *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_5th_conv_layer_call_and_return_conditional_losses_106079w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????22 `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????22 : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????22 
 
_user_specified_nameinputs
?	
?
6__inference_batch_normalization_4_layer_call_fn_108000

inputs
unknown: 
	unknown_0: 
	unknown_1: 
	unknown_2: 
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+??????????????????????????? *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Z
fURS
Q__inference_batch_normalization_4_layer_call_and_return_conditional_losses_105947?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+??????????????????????????? `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+??????????????????????????? : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs
?
c
E__inference_dropout_2_layer_call_and_return_conditional_losses_108308

inputs

identity_1N
IdentityIdentityinputs*
T0*'
_output_shapes
:?????????([

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:?????????("!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:?????????(:O K
'
_output_shapes
:?????????(
 
_user_specified_nameinputs
?
c
*__inference_dropout_2_layer_call_fn_108303

inputs
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????(* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dropout_2_layer_call_and_return_conditional_losses_106539o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????(`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:?????????(22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????(
 
_user_specified_nameinputs
?

?
D__inference_output_0_layer_call_and_return_conditional_losses_106261

inputs0
matmul_readvariableop_resource:($-
biasadd_readvariableop_resource:$
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:($*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????$r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:$*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????$V
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:?????????$`
IdentityIdentitySoftmax:softmax:0^NoOp*
T0*'
_output_shapes
:?????????$w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????(: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????(
 
_user_specified_nameinputs
?

?
T__inference_1st_fully_connected_of_4_layer_call_and_return_conditional_losses_106141

inputs1
matmul_readvariableop_resource:	?$(-
biasadd_readvariableop_resource:(
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?$(*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????(r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:(*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????(P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:?????????(a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:?????????(w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????$: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????$
 
_user_specified_nameinputs
??
?J
"__inference__traced_restore_109173
file_prefix:
 assignvariableop_4th_conv_kernel: .
 assignvariableop_1_4th_conv_bias: <
.assignvariableop_2_batch_normalization_3_gamma: ;
-assignvariableop_3_batch_normalization_3_beta: B
4assignvariableop_4_batch_normalization_3_moving_mean: F
8assignvariableop_5_batch_normalization_3_moving_variance: <
"assignvariableop_6_5th_conv_kernel:  .
 assignvariableop_7_5th_conv_bias: <
.assignvariableop_8_batch_normalization_4_gamma: ;
-assignvariableop_9_batch_normalization_4_beta: C
5assignvariableop_10_batch_normalization_4_moving_mean: G
9assignvariableop_11_batch_normalization_4_moving_variance: =
#assignvariableop_12_6th_conv_kernel:  /
!assignvariableop_13_6th_conv_bias: =
/assignvariableop_14_batch_normalization_5_gamma: <
.assignvariableop_15_batch_normalization_5_beta: C
5assignvariableop_16_batch_normalization_5_moving_mean: G
9assignvariableop_17_batch_normalization_5_moving_variance: F
3assignvariableop_18_1st_fully_connected_of_0_kernel:	?$(?
1assignvariableop_19_1st_fully_connected_of_0_bias:(F
3assignvariableop_20_1st_fully_connected_of_1_kernel:	?$(?
1assignvariableop_21_1st_fully_connected_of_1_bias:(F
3assignvariableop_22_1st_fully_connected_of_2_kernel:	?$(?
1assignvariableop_23_1st_fully_connected_of_2_bias:(F
3assignvariableop_24_1st_fully_connected_of_3_kernel:	?$(?
1assignvariableop_25_1st_fully_connected_of_3_bias:(F
3assignvariableop_26_1st_fully_connected_of_4_kernel:	?$(?
1assignvariableop_27_1st_fully_connected_of_4_bias:(5
#assignvariableop_28_output_0_kernel:($/
!assignvariableop_29_output_0_bias:$5
#assignvariableop_30_output_1_kernel:($/
!assignvariableop_31_output_1_bias:$5
#assignvariableop_32_output_2_kernel:($/
!assignvariableop_33_output_2_bias:$5
#assignvariableop_34_output_3_kernel:($/
!assignvariableop_35_output_3_bias:$5
#assignvariableop_36_output_4_kernel:($/
!assignvariableop_37_output_4_bias:$'
assignvariableop_38_adam_iter:	 )
assignvariableop_39_adam_beta_1: )
assignvariableop_40_adam_beta_2: (
assignvariableop_41_adam_decay: 0
&assignvariableop_42_adam_learning_rate: %
assignvariableop_43_total_1: %
assignvariableop_44_count_1: #
assignvariableop_45_total: #
assignvariableop_46_count: D
*assignvariableop_47_adam_4th_conv_kernel_m: 6
(assignvariableop_48_adam_4th_conv_bias_m: D
6assignvariableop_49_adam_batch_normalization_3_gamma_m: C
5assignvariableop_50_adam_batch_normalization_3_beta_m: D
*assignvariableop_51_adam_5th_conv_kernel_m:  6
(assignvariableop_52_adam_5th_conv_bias_m: D
6assignvariableop_53_adam_batch_normalization_4_gamma_m: C
5assignvariableop_54_adam_batch_normalization_4_beta_m: D
*assignvariableop_55_adam_6th_conv_kernel_m:  6
(assignvariableop_56_adam_6th_conv_bias_m: D
6assignvariableop_57_adam_batch_normalization_5_gamma_m: C
5assignvariableop_58_adam_batch_normalization_5_beta_m: M
:assignvariableop_59_adam_1st_fully_connected_of_0_kernel_m:	?$(F
8assignvariableop_60_adam_1st_fully_connected_of_0_bias_m:(M
:assignvariableop_61_adam_1st_fully_connected_of_1_kernel_m:	?$(F
8assignvariableop_62_adam_1st_fully_connected_of_1_bias_m:(M
:assignvariableop_63_adam_1st_fully_connected_of_2_kernel_m:	?$(F
8assignvariableop_64_adam_1st_fully_connected_of_2_bias_m:(M
:assignvariableop_65_adam_1st_fully_connected_of_3_kernel_m:	?$(F
8assignvariableop_66_adam_1st_fully_connected_of_3_bias_m:(M
:assignvariableop_67_adam_1st_fully_connected_of_4_kernel_m:	?$(F
8assignvariableop_68_adam_1st_fully_connected_of_4_bias_m:(<
*assignvariableop_69_adam_output_0_kernel_m:($6
(assignvariableop_70_adam_output_0_bias_m:$<
*assignvariableop_71_adam_output_1_kernel_m:($6
(assignvariableop_72_adam_output_1_bias_m:$<
*assignvariableop_73_adam_output_2_kernel_m:($6
(assignvariableop_74_adam_output_2_bias_m:$<
*assignvariableop_75_adam_output_3_kernel_m:($6
(assignvariableop_76_adam_output_3_bias_m:$<
*assignvariableop_77_adam_output_4_kernel_m:($6
(assignvariableop_78_adam_output_4_bias_m:$D
*assignvariableop_79_adam_4th_conv_kernel_v: 6
(assignvariableop_80_adam_4th_conv_bias_v: D
6assignvariableop_81_adam_batch_normalization_3_gamma_v: C
5assignvariableop_82_adam_batch_normalization_3_beta_v: D
*assignvariableop_83_adam_5th_conv_kernel_v:  6
(assignvariableop_84_adam_5th_conv_bias_v: D
6assignvariableop_85_adam_batch_normalization_4_gamma_v: C
5assignvariableop_86_adam_batch_normalization_4_beta_v: D
*assignvariableop_87_adam_6th_conv_kernel_v:  6
(assignvariableop_88_adam_6th_conv_bias_v: D
6assignvariableop_89_adam_batch_normalization_5_gamma_v: C
5assignvariableop_90_adam_batch_normalization_5_beta_v: M
:assignvariableop_91_adam_1st_fully_connected_of_0_kernel_v:	?$(F
8assignvariableop_92_adam_1st_fully_connected_of_0_bias_v:(M
:assignvariableop_93_adam_1st_fully_connected_of_1_kernel_v:	?$(F
8assignvariableop_94_adam_1st_fully_connected_of_1_bias_v:(M
:assignvariableop_95_adam_1st_fully_connected_of_2_kernel_v:	?$(F
8assignvariableop_96_adam_1st_fully_connected_of_2_bias_v:(M
:assignvariableop_97_adam_1st_fully_connected_of_3_kernel_v:	?$(F
8assignvariableop_98_adam_1st_fully_connected_of_3_bias_v:(M
:assignvariableop_99_adam_1st_fully_connected_of_4_kernel_v:	?$(G
9assignvariableop_100_adam_1st_fully_connected_of_4_bias_v:(=
+assignvariableop_101_adam_output_0_kernel_v:($7
)assignvariableop_102_adam_output_0_bias_v:$=
+assignvariableop_103_adam_output_1_kernel_v:($7
)assignvariableop_104_adam_output_1_bias_v:$=
+assignvariableop_105_adam_output_2_kernel_v:($7
)assignvariableop_106_adam_output_2_bias_v:$=
+assignvariableop_107_adam_output_3_kernel_v:($7
)assignvariableop_108_adam_output_3_bias_v:$=
+assignvariableop_109_adam_output_4_kernel_v:($7
)assignvariableop_110_adam_output_4_bias_v:$
identity_112??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_10?AssignVariableOp_100?AssignVariableOp_101?AssignVariableOp_102?AssignVariableOp_103?AssignVariableOp_104?AssignVariableOp_105?AssignVariableOp_106?AssignVariableOp_107?AssignVariableOp_108?AssignVariableOp_109?AssignVariableOp_11?AssignVariableOp_110?AssignVariableOp_12?AssignVariableOp_13?AssignVariableOp_14?AssignVariableOp_15?AssignVariableOp_16?AssignVariableOp_17?AssignVariableOp_18?AssignVariableOp_19?AssignVariableOp_2?AssignVariableOp_20?AssignVariableOp_21?AssignVariableOp_22?AssignVariableOp_23?AssignVariableOp_24?AssignVariableOp_25?AssignVariableOp_26?AssignVariableOp_27?AssignVariableOp_28?AssignVariableOp_29?AssignVariableOp_3?AssignVariableOp_30?AssignVariableOp_31?AssignVariableOp_32?AssignVariableOp_33?AssignVariableOp_34?AssignVariableOp_35?AssignVariableOp_36?AssignVariableOp_37?AssignVariableOp_38?AssignVariableOp_39?AssignVariableOp_4?AssignVariableOp_40?AssignVariableOp_41?AssignVariableOp_42?AssignVariableOp_43?AssignVariableOp_44?AssignVariableOp_45?AssignVariableOp_46?AssignVariableOp_47?AssignVariableOp_48?AssignVariableOp_49?AssignVariableOp_5?AssignVariableOp_50?AssignVariableOp_51?AssignVariableOp_52?AssignVariableOp_53?AssignVariableOp_54?AssignVariableOp_55?AssignVariableOp_56?AssignVariableOp_57?AssignVariableOp_58?AssignVariableOp_59?AssignVariableOp_6?AssignVariableOp_60?AssignVariableOp_61?AssignVariableOp_62?AssignVariableOp_63?AssignVariableOp_64?AssignVariableOp_65?AssignVariableOp_66?AssignVariableOp_67?AssignVariableOp_68?AssignVariableOp_69?AssignVariableOp_7?AssignVariableOp_70?AssignVariableOp_71?AssignVariableOp_72?AssignVariableOp_73?AssignVariableOp_74?AssignVariableOp_75?AssignVariableOp_76?AssignVariableOp_77?AssignVariableOp_78?AssignVariableOp_79?AssignVariableOp_8?AssignVariableOp_80?AssignVariableOp_81?AssignVariableOp_82?AssignVariableOp_83?AssignVariableOp_84?AssignVariableOp_85?AssignVariableOp_86?AssignVariableOp_87?AssignVariableOp_88?AssignVariableOp_89?AssignVariableOp_9?AssignVariableOp_90?AssignVariableOp_91?AssignVariableOp_92?AssignVariableOp_93?AssignVariableOp_94?AssignVariableOp_95?AssignVariableOp_96?AssignVariableOp_97?AssignVariableOp_98?AssignVariableOp_99?>
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:p*
dtype0*?>
value?>B?>pB6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-1/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-1/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-3/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-3/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-5/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-5/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-5/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-9/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-9/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-10/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-10/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-11/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-11/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-12/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-12/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-13/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-13/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-14/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-14/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-15/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-15/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-5/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-10/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-10/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-11/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-11/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-12/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-12/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-13/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-13/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-14/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-14/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-15/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-15/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-5/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-10/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-10/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-11/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-11/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-12/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-12/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-13/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-13/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-14/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-14/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-15/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-15/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH?
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:p*
dtype0*?
value?B?pB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B ?
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*?
_output_shapes?
?::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::*~
dtypest
r2p	[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOpAssignVariableOp assignvariableop_4th_conv_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_1AssignVariableOp assignvariableop_1_4th_conv_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_2AssignVariableOp.assignvariableop_2_batch_normalization_3_gammaIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_3AssignVariableOp-assignvariableop_3_batch_normalization_3_betaIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_4AssignVariableOp4assignvariableop_4_batch_normalization_3_moving_meanIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_5AssignVariableOp8assignvariableop_5_batch_normalization_3_moving_varianceIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_6AssignVariableOp"assignvariableop_6_5th_conv_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_7AssignVariableOp assignvariableop_7_5th_conv_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_8AssignVariableOp.assignvariableop_8_batch_normalization_4_gammaIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_9AssignVariableOp-assignvariableop_9_batch_normalization_4_betaIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_10AssignVariableOp5assignvariableop_10_batch_normalization_4_moving_meanIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_11AssignVariableOp9assignvariableop_11_batch_normalization_4_moving_varianceIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_12AssignVariableOp#assignvariableop_12_6th_conv_kernelIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_13AssignVariableOp!assignvariableop_13_6th_conv_biasIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_14AssignVariableOp/assignvariableop_14_batch_normalization_5_gammaIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_15AssignVariableOp.assignvariableop_15_batch_normalization_5_betaIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_16AssignVariableOp5assignvariableop_16_batch_normalization_5_moving_meanIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_17AssignVariableOp9assignvariableop_17_batch_normalization_5_moving_varianceIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_18AssignVariableOp3assignvariableop_18_1st_fully_connected_of_0_kernelIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_19AssignVariableOp1assignvariableop_19_1st_fully_connected_of_0_biasIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_20AssignVariableOp3assignvariableop_20_1st_fully_connected_of_1_kernelIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_21AssignVariableOp1assignvariableop_21_1st_fully_connected_of_1_biasIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_22AssignVariableOp3assignvariableop_22_1st_fully_connected_of_2_kernelIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_23AssignVariableOp1assignvariableop_23_1st_fully_connected_of_2_biasIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_24AssignVariableOp3assignvariableop_24_1st_fully_connected_of_3_kernelIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_25AssignVariableOp1assignvariableop_25_1st_fully_connected_of_3_biasIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_26AssignVariableOp3assignvariableop_26_1st_fully_connected_of_4_kernelIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_27AssignVariableOp1assignvariableop_27_1st_fully_connected_of_4_biasIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_28AssignVariableOp#assignvariableop_28_output_0_kernelIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_29AssignVariableOp!assignvariableop_29_output_0_biasIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_30AssignVariableOp#assignvariableop_30_output_1_kernelIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_31AssignVariableOp!assignvariableop_31_output_1_biasIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_32AssignVariableOp#assignvariableop_32_output_2_kernelIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_33AssignVariableOp!assignvariableop_33_output_2_biasIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_34AssignVariableOp#assignvariableop_34_output_3_kernelIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_35AssignVariableOp!assignvariableop_35_output_3_biasIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_36AssignVariableOp#assignvariableop_36_output_4_kernelIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_37AssignVariableOp!assignvariableop_37_output_4_biasIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0	*
_output_shapes
:?
AssignVariableOp_38AssignVariableOpassignvariableop_38_adam_iterIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	_
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_39AssignVariableOpassignvariableop_39_adam_beta_1Identity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_40AssignVariableOpassignvariableop_40_adam_beta_2Identity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_41AssignVariableOpassignvariableop_41_adam_decayIdentity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_42AssignVariableOp&assignvariableop_42_adam_learning_rateIdentity_42:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_43AssignVariableOpassignvariableop_43_total_1Identity_43:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_44AssignVariableOpassignvariableop_44_count_1Identity_44:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_45AssignVariableOpassignvariableop_45_totalIdentity_45:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_46IdentityRestoreV2:tensors:46"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_46AssignVariableOpassignvariableop_46_countIdentity_46:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_47IdentityRestoreV2:tensors:47"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_47AssignVariableOp*assignvariableop_47_adam_4th_conv_kernel_mIdentity_47:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_48IdentityRestoreV2:tensors:48"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_48AssignVariableOp(assignvariableop_48_adam_4th_conv_bias_mIdentity_48:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_49IdentityRestoreV2:tensors:49"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_49AssignVariableOp6assignvariableop_49_adam_batch_normalization_3_gamma_mIdentity_49:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_50IdentityRestoreV2:tensors:50"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_50AssignVariableOp5assignvariableop_50_adam_batch_normalization_3_beta_mIdentity_50:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_51IdentityRestoreV2:tensors:51"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_51AssignVariableOp*assignvariableop_51_adam_5th_conv_kernel_mIdentity_51:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_52IdentityRestoreV2:tensors:52"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_52AssignVariableOp(assignvariableop_52_adam_5th_conv_bias_mIdentity_52:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_53IdentityRestoreV2:tensors:53"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_53AssignVariableOp6assignvariableop_53_adam_batch_normalization_4_gamma_mIdentity_53:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_54IdentityRestoreV2:tensors:54"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_54AssignVariableOp5assignvariableop_54_adam_batch_normalization_4_beta_mIdentity_54:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_55IdentityRestoreV2:tensors:55"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_55AssignVariableOp*assignvariableop_55_adam_6th_conv_kernel_mIdentity_55:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_56IdentityRestoreV2:tensors:56"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_56AssignVariableOp(assignvariableop_56_adam_6th_conv_bias_mIdentity_56:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_57IdentityRestoreV2:tensors:57"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_57AssignVariableOp6assignvariableop_57_adam_batch_normalization_5_gamma_mIdentity_57:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_58IdentityRestoreV2:tensors:58"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_58AssignVariableOp5assignvariableop_58_adam_batch_normalization_5_beta_mIdentity_58:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_59IdentityRestoreV2:tensors:59"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_59AssignVariableOp:assignvariableop_59_adam_1st_fully_connected_of_0_kernel_mIdentity_59:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_60IdentityRestoreV2:tensors:60"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_60AssignVariableOp8assignvariableop_60_adam_1st_fully_connected_of_0_bias_mIdentity_60:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_61IdentityRestoreV2:tensors:61"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_61AssignVariableOp:assignvariableop_61_adam_1st_fully_connected_of_1_kernel_mIdentity_61:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_62IdentityRestoreV2:tensors:62"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_62AssignVariableOp8assignvariableop_62_adam_1st_fully_connected_of_1_bias_mIdentity_62:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_63IdentityRestoreV2:tensors:63"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_63AssignVariableOp:assignvariableop_63_adam_1st_fully_connected_of_2_kernel_mIdentity_63:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_64IdentityRestoreV2:tensors:64"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_64AssignVariableOp8assignvariableop_64_adam_1st_fully_connected_of_2_bias_mIdentity_64:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_65IdentityRestoreV2:tensors:65"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_65AssignVariableOp:assignvariableop_65_adam_1st_fully_connected_of_3_kernel_mIdentity_65:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_66IdentityRestoreV2:tensors:66"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_66AssignVariableOp8assignvariableop_66_adam_1st_fully_connected_of_3_bias_mIdentity_66:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_67IdentityRestoreV2:tensors:67"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_67AssignVariableOp:assignvariableop_67_adam_1st_fully_connected_of_4_kernel_mIdentity_67:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_68IdentityRestoreV2:tensors:68"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_68AssignVariableOp8assignvariableop_68_adam_1st_fully_connected_of_4_bias_mIdentity_68:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_69IdentityRestoreV2:tensors:69"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_69AssignVariableOp*assignvariableop_69_adam_output_0_kernel_mIdentity_69:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_70IdentityRestoreV2:tensors:70"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_70AssignVariableOp(assignvariableop_70_adam_output_0_bias_mIdentity_70:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_71IdentityRestoreV2:tensors:71"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_71AssignVariableOp*assignvariableop_71_adam_output_1_kernel_mIdentity_71:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_72IdentityRestoreV2:tensors:72"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_72AssignVariableOp(assignvariableop_72_adam_output_1_bias_mIdentity_72:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_73IdentityRestoreV2:tensors:73"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_73AssignVariableOp*assignvariableop_73_adam_output_2_kernel_mIdentity_73:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_74IdentityRestoreV2:tensors:74"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_74AssignVariableOp(assignvariableop_74_adam_output_2_bias_mIdentity_74:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_75IdentityRestoreV2:tensors:75"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_75AssignVariableOp*assignvariableop_75_adam_output_3_kernel_mIdentity_75:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_76IdentityRestoreV2:tensors:76"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_76AssignVariableOp(assignvariableop_76_adam_output_3_bias_mIdentity_76:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_77IdentityRestoreV2:tensors:77"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_77AssignVariableOp*assignvariableop_77_adam_output_4_kernel_mIdentity_77:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_78IdentityRestoreV2:tensors:78"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_78AssignVariableOp(assignvariableop_78_adam_output_4_bias_mIdentity_78:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_79IdentityRestoreV2:tensors:79"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_79AssignVariableOp*assignvariableop_79_adam_4th_conv_kernel_vIdentity_79:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_80IdentityRestoreV2:tensors:80"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_80AssignVariableOp(assignvariableop_80_adam_4th_conv_bias_vIdentity_80:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_81IdentityRestoreV2:tensors:81"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_81AssignVariableOp6assignvariableop_81_adam_batch_normalization_3_gamma_vIdentity_81:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_82IdentityRestoreV2:tensors:82"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_82AssignVariableOp5assignvariableop_82_adam_batch_normalization_3_beta_vIdentity_82:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_83IdentityRestoreV2:tensors:83"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_83AssignVariableOp*assignvariableop_83_adam_5th_conv_kernel_vIdentity_83:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_84IdentityRestoreV2:tensors:84"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_84AssignVariableOp(assignvariableop_84_adam_5th_conv_bias_vIdentity_84:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_85IdentityRestoreV2:tensors:85"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_85AssignVariableOp6assignvariableop_85_adam_batch_normalization_4_gamma_vIdentity_85:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_86IdentityRestoreV2:tensors:86"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_86AssignVariableOp5assignvariableop_86_adam_batch_normalization_4_beta_vIdentity_86:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_87IdentityRestoreV2:tensors:87"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_87AssignVariableOp*assignvariableop_87_adam_6th_conv_kernel_vIdentity_87:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_88IdentityRestoreV2:tensors:88"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_88AssignVariableOp(assignvariableop_88_adam_6th_conv_bias_vIdentity_88:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_89IdentityRestoreV2:tensors:89"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_89AssignVariableOp6assignvariableop_89_adam_batch_normalization_5_gamma_vIdentity_89:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_90IdentityRestoreV2:tensors:90"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_90AssignVariableOp5assignvariableop_90_adam_batch_normalization_5_beta_vIdentity_90:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_91IdentityRestoreV2:tensors:91"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_91AssignVariableOp:assignvariableop_91_adam_1st_fully_connected_of_0_kernel_vIdentity_91:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_92IdentityRestoreV2:tensors:92"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_92AssignVariableOp8assignvariableop_92_adam_1st_fully_connected_of_0_bias_vIdentity_92:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_93IdentityRestoreV2:tensors:93"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_93AssignVariableOp:assignvariableop_93_adam_1st_fully_connected_of_1_kernel_vIdentity_93:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_94IdentityRestoreV2:tensors:94"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_94AssignVariableOp8assignvariableop_94_adam_1st_fully_connected_of_1_bias_vIdentity_94:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_95IdentityRestoreV2:tensors:95"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_95AssignVariableOp:assignvariableop_95_adam_1st_fully_connected_of_2_kernel_vIdentity_95:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_96IdentityRestoreV2:tensors:96"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_96AssignVariableOp8assignvariableop_96_adam_1st_fully_connected_of_2_bias_vIdentity_96:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_97IdentityRestoreV2:tensors:97"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_97AssignVariableOp:assignvariableop_97_adam_1st_fully_connected_of_3_kernel_vIdentity_97:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_98IdentityRestoreV2:tensors:98"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_98AssignVariableOp8assignvariableop_98_adam_1st_fully_connected_of_3_bias_vIdentity_98:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_99IdentityRestoreV2:tensors:99"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_99AssignVariableOp:assignvariableop_99_adam_1st_fully_connected_of_4_kernel_vIdentity_99:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_100IdentityRestoreV2:tensors:100"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_100AssignVariableOp9assignvariableop_100_adam_1st_fully_connected_of_4_bias_vIdentity_100:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_101IdentityRestoreV2:tensors:101"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_101AssignVariableOp+assignvariableop_101_adam_output_0_kernel_vIdentity_101:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_102IdentityRestoreV2:tensors:102"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_102AssignVariableOp)assignvariableop_102_adam_output_0_bias_vIdentity_102:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_103IdentityRestoreV2:tensors:103"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_103AssignVariableOp+assignvariableop_103_adam_output_1_kernel_vIdentity_103:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_104IdentityRestoreV2:tensors:104"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_104AssignVariableOp)assignvariableop_104_adam_output_1_bias_vIdentity_104:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_105IdentityRestoreV2:tensors:105"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_105AssignVariableOp+assignvariableop_105_adam_output_2_kernel_vIdentity_105:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_106IdentityRestoreV2:tensors:106"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_106AssignVariableOp)assignvariableop_106_adam_output_2_bias_vIdentity_106:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_107IdentityRestoreV2:tensors:107"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_107AssignVariableOp+assignvariableop_107_adam_output_3_kernel_vIdentity_107:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_108IdentityRestoreV2:tensors:108"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_108AssignVariableOp)assignvariableop_108_adam_output_3_bias_vIdentity_108:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_109IdentityRestoreV2:tensors:109"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_109AssignVariableOp+assignvariableop_109_adam_output_4_kernel_vIdentity_109:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_110IdentityRestoreV2:tensors:110"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_110AssignVariableOp)assignvariableop_110_adam_output_4_bias_vIdentity_110:output:0"/device:CPU:0*
_output_shapes
 *
dtype01
NoOpNoOp"/device:CPU:0*
_output_shapes
 ?
Identity_111Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_100^AssignVariableOp_101^AssignVariableOp_102^AssignVariableOp_103^AssignVariableOp_104^AssignVariableOp_105^AssignVariableOp_106^AssignVariableOp_107^AssignVariableOp_108^AssignVariableOp_109^AssignVariableOp_11^AssignVariableOp_110^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_69^AssignVariableOp_7^AssignVariableOp_70^AssignVariableOp_71^AssignVariableOp_72^AssignVariableOp_73^AssignVariableOp_74^AssignVariableOp_75^AssignVariableOp_76^AssignVariableOp_77^AssignVariableOp_78^AssignVariableOp_79^AssignVariableOp_8^AssignVariableOp_80^AssignVariableOp_81^AssignVariableOp_82^AssignVariableOp_83^AssignVariableOp_84^AssignVariableOp_85^AssignVariableOp_86^AssignVariableOp_87^AssignVariableOp_88^AssignVariableOp_89^AssignVariableOp_9^AssignVariableOp_90^AssignVariableOp_91^AssignVariableOp_92^AssignVariableOp_93^AssignVariableOp_94^AssignVariableOp_95^AssignVariableOp_96^AssignVariableOp_97^AssignVariableOp_98^AssignVariableOp_99^NoOp"/device:CPU:0*
T0*
_output_shapes
: Y
Identity_112IdentityIdentity_111:output:0^NoOp_1*
T0*
_output_shapes
: ?
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_100^AssignVariableOp_101^AssignVariableOp_102^AssignVariableOp_103^AssignVariableOp_104^AssignVariableOp_105^AssignVariableOp_106^AssignVariableOp_107^AssignVariableOp_108^AssignVariableOp_109^AssignVariableOp_11^AssignVariableOp_110^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_69^AssignVariableOp_7^AssignVariableOp_70^AssignVariableOp_71^AssignVariableOp_72^AssignVariableOp_73^AssignVariableOp_74^AssignVariableOp_75^AssignVariableOp_76^AssignVariableOp_77^AssignVariableOp_78^AssignVariableOp_79^AssignVariableOp_8^AssignVariableOp_80^AssignVariableOp_81^AssignVariableOp_82^AssignVariableOp_83^AssignVariableOp_84^AssignVariableOp_85^AssignVariableOp_86^AssignVariableOp_87^AssignVariableOp_88^AssignVariableOp_89^AssignVariableOp_9^AssignVariableOp_90^AssignVariableOp_91^AssignVariableOp_92^AssignVariableOp_93^AssignVariableOp_94^AssignVariableOp_95^AssignVariableOp_96^AssignVariableOp_97^AssignVariableOp_98^AssignVariableOp_99*"
_acd_function_control_output(*
_output_shapes
 "%
identity_112Identity_112:output:0*?
_input_shapes?
?: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102,
AssignVariableOp_100AssignVariableOp_1002,
AssignVariableOp_101AssignVariableOp_1012,
AssignVariableOp_102AssignVariableOp_1022,
AssignVariableOp_103AssignVariableOp_1032,
AssignVariableOp_104AssignVariableOp_1042,
AssignVariableOp_105AssignVariableOp_1052,
AssignVariableOp_106AssignVariableOp_1062,
AssignVariableOp_107AssignVariableOp_1072,
AssignVariableOp_108AssignVariableOp_1082,
AssignVariableOp_109AssignVariableOp_1092*
AssignVariableOp_11AssignVariableOp_112,
AssignVariableOp_110AssignVariableOp_1102*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_34AssignVariableOp_342*
AssignVariableOp_35AssignVariableOp_352*
AssignVariableOp_36AssignVariableOp_362*
AssignVariableOp_37AssignVariableOp_372*
AssignVariableOp_38AssignVariableOp_382*
AssignVariableOp_39AssignVariableOp_392(
AssignVariableOp_4AssignVariableOp_42*
AssignVariableOp_40AssignVariableOp_402*
AssignVariableOp_41AssignVariableOp_412*
AssignVariableOp_42AssignVariableOp_422*
AssignVariableOp_43AssignVariableOp_432*
AssignVariableOp_44AssignVariableOp_442*
AssignVariableOp_45AssignVariableOp_452*
AssignVariableOp_46AssignVariableOp_462*
AssignVariableOp_47AssignVariableOp_472*
AssignVariableOp_48AssignVariableOp_482*
AssignVariableOp_49AssignVariableOp_492(
AssignVariableOp_5AssignVariableOp_52*
AssignVariableOp_50AssignVariableOp_502*
AssignVariableOp_51AssignVariableOp_512*
AssignVariableOp_52AssignVariableOp_522*
AssignVariableOp_53AssignVariableOp_532*
AssignVariableOp_54AssignVariableOp_542*
AssignVariableOp_55AssignVariableOp_552*
AssignVariableOp_56AssignVariableOp_562*
AssignVariableOp_57AssignVariableOp_572*
AssignVariableOp_58AssignVariableOp_582*
AssignVariableOp_59AssignVariableOp_592(
AssignVariableOp_6AssignVariableOp_62*
AssignVariableOp_60AssignVariableOp_602*
AssignVariableOp_61AssignVariableOp_612*
AssignVariableOp_62AssignVariableOp_622*
AssignVariableOp_63AssignVariableOp_632*
AssignVariableOp_64AssignVariableOp_642*
AssignVariableOp_65AssignVariableOp_652*
AssignVariableOp_66AssignVariableOp_662*
AssignVariableOp_67AssignVariableOp_672*
AssignVariableOp_68AssignVariableOp_682*
AssignVariableOp_69AssignVariableOp_692(
AssignVariableOp_7AssignVariableOp_72*
AssignVariableOp_70AssignVariableOp_702*
AssignVariableOp_71AssignVariableOp_712*
AssignVariableOp_72AssignVariableOp_722*
AssignVariableOp_73AssignVariableOp_732*
AssignVariableOp_74AssignVariableOp_742*
AssignVariableOp_75AssignVariableOp_752*
AssignVariableOp_76AssignVariableOp_762*
AssignVariableOp_77AssignVariableOp_772*
AssignVariableOp_78AssignVariableOp_782*
AssignVariableOp_79AssignVariableOp_792(
AssignVariableOp_8AssignVariableOp_82*
AssignVariableOp_80AssignVariableOp_802*
AssignVariableOp_81AssignVariableOp_812*
AssignVariableOp_82AssignVariableOp_822*
AssignVariableOp_83AssignVariableOp_832*
AssignVariableOp_84AssignVariableOp_842*
AssignVariableOp_85AssignVariableOp_852*
AssignVariableOp_86AssignVariableOp_862*
AssignVariableOp_87AssignVariableOp_872*
AssignVariableOp_88AssignVariableOp_882*
AssignVariableOp_89AssignVariableOp_892(
AssignVariableOp_9AssignVariableOp_92*
AssignVariableOp_90AssignVariableOp_902*
AssignVariableOp_91AssignVariableOp_912*
AssignVariableOp_92AssignVariableOp_922*
AssignVariableOp_93AssignVariableOp_932*
AssignVariableOp_94AssignVariableOp_942*
AssignVariableOp_95AssignVariableOp_952*
AssignVariableOp_96AssignVariableOp_962*
AssignVariableOp_97AssignVariableOp_972*
AssignVariableOp_98AssignVariableOp_982*
AssignVariableOp_99AssignVariableOp_99:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
?
?
&__inference_model_layer_call_fn_107030

images!
unknown: 
	unknown_0: 
	unknown_1: 
	unknown_2: 
	unknown_3: 
	unknown_4: #
	unknown_5:  
	unknown_6: 
	unknown_7: 
	unknown_8: 
	unknown_9: 

unknown_10: $

unknown_11:  

unknown_12: 

unknown_13: 

unknown_14: 

unknown_15: 

unknown_16: 

unknown_17:	?$(

unknown_18:(

unknown_19:	?$(

unknown_20:(

unknown_21:	?$(

unknown_22:(

unknown_23:	?$(

unknown_24:(

unknown_25:	?$(

unknown_26:(

unknown_27:($

unknown_28:$

unknown_29:($

unknown_30:$

unknown_31:($

unknown_32:$

unknown_33:($

unknown_34:$

unknown_35:($

unknown_36:$
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallimagesunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36*2
Tin+
)2'*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????$*B
_read_only_resource_inputs$
" 	
 !"#$%&*-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_model_layer_call_and_return_conditional_losses_106870s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:?????????$`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*z
_input_shapesi
g:?????????dd: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????dd
 
_user_specified_nameimages
?

?
T__inference_1st_fully_connected_of_3_layer_call_and_return_conditional_losses_106158

inputs1
matmul_readvariableop_resource:	?$(-
biasadd_readvariableop_resource:(
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?$(*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????(r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:(*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????(P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:?????????(a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:?????????(w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????$: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????$
 
_user_specified_nameinputs
?
c
E__inference_dropout_4_layer_call_and_return_conditional_losses_106220

inputs

identity_1N
IdentityIdentityinputs*
T0*'
_output_shapes
:?????????([

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:?????????("!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:?????????(:O K
'
_output_shapes
:?????????(
 
_user_specified_nameinputs
??
?!
A__inference_model_layer_call_and_return_conditional_losses_107662

inputs@
&th_conv_conv2d_readvariableop_resource: 5
'th_conv_biasadd_readvariableop_resource: ;
-batch_normalization_3_readvariableop_resource: =
/batch_normalization_3_readvariableop_1_resource: L
>batch_normalization_3_fusedbatchnormv3_readvariableop_resource: N
@batch_normalization_3_fusedbatchnormv3_readvariableop_1_resource: B
(th_conv_conv2d_readvariableop_resource_0:  7
)th_conv_biasadd_readvariableop_resource_0: ;
-batch_normalization_4_readvariableop_resource: =
/batch_normalization_4_readvariableop_1_resource: L
>batch_normalization_4_fusedbatchnormv3_readvariableop_resource: N
@batch_normalization_4_fusedbatchnormv3_readvariableop_1_resource: B
(th_conv_conv2d_readvariableop_resource_1:  7
)th_conv_biasadd_readvariableop_resource_1: ;
-batch_normalization_5_readvariableop_resource: =
/batch_normalization_5_readvariableop_1_resource: L
>batch_normalization_5_fusedbatchnormv3_readvariableop_resource: N
@batch_normalization_5_fusedbatchnormv3_readvariableop_1_resource: I
6st_fully_connected_of_4_matmul_readvariableop_resource:	?$(E
7st_fully_connected_of_4_biasadd_readvariableop_resource:(I
6st_fully_connected_of_3_matmul_readvariableop_resource:	?$(E
7st_fully_connected_of_3_biasadd_readvariableop_resource:(I
6st_fully_connected_of_2_matmul_readvariableop_resource:	?$(E
7st_fully_connected_of_2_biasadd_readvariableop_resource:(I
6st_fully_connected_of_1_matmul_readvariableop_resource:	?$(E
7st_fully_connected_of_1_biasadd_readvariableop_resource:(I
6st_fully_connected_of_0_matmul_readvariableop_resource:	?$(E
7st_fully_connected_of_0_biasadd_readvariableop_resource:(9
'output_0_matmul_readvariableop_resource:($6
(output_0_biasadd_readvariableop_resource:$9
'output_1_matmul_readvariableop_resource:($6
(output_1_biasadd_readvariableop_resource:$9
'output_2_matmul_readvariableop_resource:($6
(output_2_biasadd_readvariableop_resource:$9
'output_3_matmul_readvariableop_resource:($6
(output_3_biasadd_readvariableop_resource:$9
'output_4_matmul_readvariableop_resource:($6
(output_4_biasadd_readvariableop_resource:$
identity??/1st_fully_connected_of_0/BiasAdd/ReadVariableOp?.1st_fully_connected_of_0/MatMul/ReadVariableOp?/1st_fully_connected_of_1/BiasAdd/ReadVariableOp?.1st_fully_connected_of_1/MatMul/ReadVariableOp?/1st_fully_connected_of_2/BiasAdd/ReadVariableOp?.1st_fully_connected_of_2/MatMul/ReadVariableOp?/1st_fully_connected_of_3/BiasAdd/ReadVariableOp?.1st_fully_connected_of_3/MatMul/ReadVariableOp?/1st_fully_connected_of_4/BiasAdd/ReadVariableOp?.1st_fully_connected_of_4/MatMul/ReadVariableOp?4th_conv/BiasAdd/ReadVariableOp?4th_conv/Conv2D/ReadVariableOp?5th_conv/BiasAdd/ReadVariableOp?5th_conv/Conv2D/ReadVariableOp?6th_conv/BiasAdd/ReadVariableOp?6th_conv/Conv2D/ReadVariableOp?5batch_normalization_3/FusedBatchNormV3/ReadVariableOp?7batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1?$batch_normalization_3/ReadVariableOp?&batch_normalization_3/ReadVariableOp_1?5batch_normalization_4/FusedBatchNormV3/ReadVariableOp?7batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1?$batch_normalization_4/ReadVariableOp?&batch_normalization_4/ReadVariableOp_1?5batch_normalization_5/FusedBatchNormV3/ReadVariableOp?7batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1?$batch_normalization_5/ReadVariableOp?&batch_normalization_5/ReadVariableOp_1?output_0/BiasAdd/ReadVariableOp?output_0/MatMul/ReadVariableOp?output_1/BiasAdd/ReadVariableOp?output_1/MatMul/ReadVariableOp?output_2/BiasAdd/ReadVariableOp?output_2/MatMul/ReadVariableOp?output_3/BiasAdd/ReadVariableOp?output_3/MatMul/ReadVariableOp?output_4/BiasAdd/ReadVariableOp?output_4/MatMul/ReadVariableOp?
4th_conv/Conv2D/ReadVariableOpReadVariableOp&th_conv_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0?
4th_conv/Conv2DConv2Dinputs&4th_conv/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????dd *
paddingSAME*
strides
?
4th_conv/BiasAdd/ReadVariableOpReadVariableOp'th_conv_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0?
4th_conv/BiasAddBiasAdd4th_conv/Conv2D:output:0'4th_conv/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????dd j
4th_conv/ReluRelu4th_conv/BiasAdd:output:0*
T0*/
_output_shapes
:?????????dd ?
4th_pool/MaxPoolMaxPool4th_conv/Relu:activations:0*/
_output_shapes
:?????????22 *
ksize
*
paddingVALID*
strides
?
$batch_normalization_3/ReadVariableOpReadVariableOp-batch_normalization_3_readvariableop_resource*
_output_shapes
: *
dtype0?
&batch_normalization_3/ReadVariableOp_1ReadVariableOp/batch_normalization_3_readvariableop_1_resource*
_output_shapes
: *
dtype0?
5batch_normalization_3/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_3_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0?
7batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_3_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0?
&batch_normalization_3/FusedBatchNormV3FusedBatchNormV34th_pool/MaxPool:output:0,batch_normalization_3/ReadVariableOp:value:0.batch_normalization_3/ReadVariableOp_1:value:0=batch_normalization_3/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????22 : : : : :*
epsilon%o?:*
is_training( ?
5th_conv/Conv2D/ReadVariableOpReadVariableOp(th_conv_conv2d_readvariableop_resource_0*&
_output_shapes
:  *
dtype0?
5th_conv/Conv2DConv2D*batch_normalization_3/FusedBatchNormV3:y:0&5th_conv/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????22 *
paddingSAME*
strides
?
5th_conv/BiasAdd/ReadVariableOpReadVariableOp)th_conv_biasadd_readvariableop_resource_0*
_output_shapes
: *
dtype0?
5th_conv/BiasAddBiasAdd5th_conv/Conv2D:output:0'5th_conv/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????22 j
5th_conv/ReluRelu5th_conv/BiasAdd:output:0*
T0*/
_output_shapes
:?????????22 ?
5th_pool/MaxPoolMaxPool5th_conv/Relu:activations:0*/
_output_shapes
:????????? *
ksize
*
paddingVALID*
strides
?
$batch_normalization_4/ReadVariableOpReadVariableOp-batch_normalization_4_readvariableop_resource*
_output_shapes
: *
dtype0?
&batch_normalization_4/ReadVariableOp_1ReadVariableOp/batch_normalization_4_readvariableop_1_resource*
_output_shapes
: *
dtype0?
5batch_normalization_4/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_4_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0?
7batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_4_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0?
&batch_normalization_4/FusedBatchNormV3FusedBatchNormV35th_pool/MaxPool:output:0,batch_normalization_4/ReadVariableOp:value:0.batch_normalization_4/ReadVariableOp_1:value:0=batch_normalization_4/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:????????? : : : : :*
epsilon%o?:*
is_training( ?
6th_conv/Conv2D/ReadVariableOpReadVariableOp(th_conv_conv2d_readvariableop_resource_1*&
_output_shapes
:  *
dtype0?
6th_conv/Conv2DConv2D*batch_normalization_4/FusedBatchNormV3:y:0&6th_conv/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? *
paddingSAME*
strides
?
6th_conv/BiasAdd/ReadVariableOpReadVariableOp)th_conv_biasadd_readvariableop_resource_1*
_output_shapes
: *
dtype0?
6th_conv/BiasAddBiasAdd6th_conv/Conv2D:output:0'6th_conv/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? j
6th_conv/ReluRelu6th_conv/BiasAdd:output:0*
T0*/
_output_shapes
:????????? ?
6th_pool/MaxPoolMaxPool6th_conv/Relu:activations:0*/
_output_shapes
:????????? *
ksize
*
paddingVALID*
strides
?
$batch_normalization_5/ReadVariableOpReadVariableOp-batch_normalization_5_readvariableop_resource*
_output_shapes
: *
dtype0?
&batch_normalization_5/ReadVariableOp_1ReadVariableOp/batch_normalization_5_readvariableop_1_resource*
_output_shapes
: *
dtype0?
5batch_normalization_5/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_5_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0?
7batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_5_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0?
&batch_normalization_5/FusedBatchNormV3FusedBatchNormV36th_pool/MaxPool:output:0,batch_normalization_5/ReadVariableOp:value:0.batch_normalization_5/ReadVariableOp_1:value:0=batch_normalization_5/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:????????? : : : : :*
epsilon%o?:*
is_training( [

flat/ConstConst*
_output_shapes
:*
dtype0*
valueB"????   ?
flat/ReshapeReshape*batch_normalization_5/FusedBatchNormV3:y:0flat/Const:output:0*
T0*(
_output_shapes
:??????????$?
.1st_fully_connected_of_4/MatMul/ReadVariableOpReadVariableOp6st_fully_connected_of_4_matmul_readvariableop_resource*
_output_shapes
:	?$(*
dtype0?
1st_fully_connected_of_4/MatMulMatMulflat/Reshape:output:061st_fully_connected_of_4/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????(?
/1st_fully_connected_of_4/BiasAdd/ReadVariableOpReadVariableOp7st_fully_connected_of_4_biasadd_readvariableop_resource*
_output_shapes
:(*
dtype0?
 1st_fully_connected_of_4/BiasAddBiasAdd)1st_fully_connected_of_4/MatMul:product:071st_fully_connected_of_4/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????(?
1st_fully_connected_of_4/ReluRelu)1st_fully_connected_of_4/BiasAdd:output:0*
T0*'
_output_shapes
:?????????(?
.1st_fully_connected_of_3/MatMul/ReadVariableOpReadVariableOp6st_fully_connected_of_3_matmul_readvariableop_resource*
_output_shapes
:	?$(*
dtype0?
1st_fully_connected_of_3/MatMulMatMulflat/Reshape:output:061st_fully_connected_of_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????(?
/1st_fully_connected_of_3/BiasAdd/ReadVariableOpReadVariableOp7st_fully_connected_of_3_biasadd_readvariableop_resource*
_output_shapes
:(*
dtype0?
 1st_fully_connected_of_3/BiasAddBiasAdd)1st_fully_connected_of_3/MatMul:product:071st_fully_connected_of_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????(?
1st_fully_connected_of_3/ReluRelu)1st_fully_connected_of_3/BiasAdd:output:0*
T0*'
_output_shapes
:?????????(?
.1st_fully_connected_of_2/MatMul/ReadVariableOpReadVariableOp6st_fully_connected_of_2_matmul_readvariableop_resource*
_output_shapes
:	?$(*
dtype0?
1st_fully_connected_of_2/MatMulMatMulflat/Reshape:output:061st_fully_connected_of_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????(?
/1st_fully_connected_of_2/BiasAdd/ReadVariableOpReadVariableOp7st_fully_connected_of_2_biasadd_readvariableop_resource*
_output_shapes
:(*
dtype0?
 1st_fully_connected_of_2/BiasAddBiasAdd)1st_fully_connected_of_2/MatMul:product:071st_fully_connected_of_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????(?
1st_fully_connected_of_2/ReluRelu)1st_fully_connected_of_2/BiasAdd:output:0*
T0*'
_output_shapes
:?????????(?
.1st_fully_connected_of_1/MatMul/ReadVariableOpReadVariableOp6st_fully_connected_of_1_matmul_readvariableop_resource*
_output_shapes
:	?$(*
dtype0?
1st_fully_connected_of_1/MatMulMatMulflat/Reshape:output:061st_fully_connected_of_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????(?
/1st_fully_connected_of_1/BiasAdd/ReadVariableOpReadVariableOp7st_fully_connected_of_1_biasadd_readvariableop_resource*
_output_shapes
:(*
dtype0?
 1st_fully_connected_of_1/BiasAddBiasAdd)1st_fully_connected_of_1/MatMul:product:071st_fully_connected_of_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????(?
1st_fully_connected_of_1/ReluRelu)1st_fully_connected_of_1/BiasAdd:output:0*
T0*'
_output_shapes
:?????????(?
.1st_fully_connected_of_0/MatMul/ReadVariableOpReadVariableOp6st_fully_connected_of_0_matmul_readvariableop_resource*
_output_shapes
:	?$(*
dtype0?
1st_fully_connected_of_0/MatMulMatMulflat/Reshape:output:061st_fully_connected_of_0/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????(?
/1st_fully_connected_of_0/BiasAdd/ReadVariableOpReadVariableOp7st_fully_connected_of_0_biasadd_readvariableop_resource*
_output_shapes
:(*
dtype0?
 1st_fully_connected_of_0/BiasAddBiasAdd)1st_fully_connected_of_0/MatMul:product:071st_fully_connected_of_0/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????(?
1st_fully_connected_of_0/ReluRelu)1st_fully_connected_of_0/BiasAdd:output:0*
T0*'
_output_shapes
:?????????(}
dropout_4/IdentityIdentity+1st_fully_connected_of_4/Relu:activations:0*
T0*'
_output_shapes
:?????????(}
dropout_3/IdentityIdentity+1st_fully_connected_of_3/Relu:activations:0*
T0*'
_output_shapes
:?????????(}
dropout_2/IdentityIdentity+1st_fully_connected_of_2/Relu:activations:0*
T0*'
_output_shapes
:?????????(}
dropout_1/IdentityIdentity+1st_fully_connected_of_1/Relu:activations:0*
T0*'
_output_shapes
:?????????({
dropout/IdentityIdentity+1st_fully_connected_of_0/Relu:activations:0*
T0*'
_output_shapes
:?????????(?
output_0/MatMul/ReadVariableOpReadVariableOp'output_0_matmul_readvariableop_resource*
_output_shapes

:($*
dtype0?
output_0/MatMulMatMuldropout/Identity:output:0&output_0/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????$?
output_0/BiasAdd/ReadVariableOpReadVariableOp(output_0_biasadd_readvariableop_resource*
_output_shapes
:$*
dtype0?
output_0/BiasAddBiasAddoutput_0/MatMul:product:0'output_0/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????$h
output_0/SoftmaxSoftmaxoutput_0/BiasAdd:output:0*
T0*'
_output_shapes
:?????????$?
output_1/MatMul/ReadVariableOpReadVariableOp'output_1_matmul_readvariableop_resource*
_output_shapes

:($*
dtype0?
output_1/MatMulMatMuldropout_1/Identity:output:0&output_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????$?
output_1/BiasAdd/ReadVariableOpReadVariableOp(output_1_biasadd_readvariableop_resource*
_output_shapes
:$*
dtype0?
output_1/BiasAddBiasAddoutput_1/MatMul:product:0'output_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????$h
output_1/SoftmaxSoftmaxoutput_1/BiasAdd:output:0*
T0*'
_output_shapes
:?????????$?
output_2/MatMul/ReadVariableOpReadVariableOp'output_2_matmul_readvariableop_resource*
_output_shapes

:($*
dtype0?
output_2/MatMulMatMuldropout_2/Identity:output:0&output_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????$?
output_2/BiasAdd/ReadVariableOpReadVariableOp(output_2_biasadd_readvariableop_resource*
_output_shapes
:$*
dtype0?
output_2/BiasAddBiasAddoutput_2/MatMul:product:0'output_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????$h
output_2/SoftmaxSoftmaxoutput_2/BiasAdd:output:0*
T0*'
_output_shapes
:?????????$?
output_3/MatMul/ReadVariableOpReadVariableOp'output_3_matmul_readvariableop_resource*
_output_shapes

:($*
dtype0?
output_3/MatMulMatMuldropout_3/Identity:output:0&output_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????$?
output_3/BiasAdd/ReadVariableOpReadVariableOp(output_3_biasadd_readvariableop_resource*
_output_shapes
:$*
dtype0?
output_3/BiasAddBiasAddoutput_3/MatMul:product:0'output_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????$h
output_3/SoftmaxSoftmaxoutput_3/BiasAdd:output:0*
T0*'
_output_shapes
:?????????$?
output_4/MatMul/ReadVariableOpReadVariableOp'output_4_matmul_readvariableop_resource*
_output_shapes

:($*
dtype0?
output_4/MatMulMatMuldropout_4/Identity:output:0&output_4/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????$?
output_4/BiasAdd/ReadVariableOpReadVariableOp(output_4_biasadd_readvariableop_resource*
_output_shapes
:$*
dtype0?
output_4/BiasAddBiasAddoutput_4/MatMul:product:0'output_4/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????$h
output_4/SoftmaxSoftmaxoutput_4/BiasAdd:output:0*
T0*'
_output_shapes
:?????????$?
tf.convert_to_tensor_1/CastCastoutput_0/Softmax:softmax:0*

DstT0*

SrcT0*'
_output_shapes
:?????????$?
tf.convert_to_tensor_1/Cast_1Castoutput_1/Softmax:softmax:0*

DstT0*

SrcT0*'
_output_shapes
:?????????$?
tf.convert_to_tensor_1/Cast_2Castoutput_2/Softmax:softmax:0*

DstT0*

SrcT0*'
_output_shapes
:?????????$?
tf.convert_to_tensor_1/Cast_3Castoutput_3/Softmax:softmax:0*

DstT0*

SrcT0*'
_output_shapes
:?????????$?
tf.convert_to_tensor_1/Cast_4Castoutput_4/Softmax:softmax:0*

DstT0*

SrcT0*'
_output_shapes
:?????????$?
tf.convert_to_tensor_1/packedPacktf.convert_to_tensor_1/Cast:y:0!tf.convert_to_tensor_1/Cast_1:y:0!tf.convert_to_tensor_1/Cast_2:y:0!tf.convert_to_tensor_1/Cast_3:y:0!tf.convert_to_tensor_1/Cast_4:y:0*
N*
T0*+
_output_shapes
:?????????$|
'tf.compat.v1.transpose_1/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          ?
"tf.compat.v1.transpose_1/transpose	Transpose&tf.convert_to_tensor_1/packed:output:00tf.compat.v1.transpose_1/transpose/perm:output:0*
T0*+
_output_shapes
:?????????$y
IdentityIdentity&tf.compat.v1.transpose_1/transpose:y:0^NoOp*
T0*+
_output_shapes
:?????????$?
NoOpNoOp0^1st_fully_connected_of_0/BiasAdd/ReadVariableOp/^1st_fully_connected_of_0/MatMul/ReadVariableOp0^1st_fully_connected_of_1/BiasAdd/ReadVariableOp/^1st_fully_connected_of_1/MatMul/ReadVariableOp0^1st_fully_connected_of_2/BiasAdd/ReadVariableOp/^1st_fully_connected_of_2/MatMul/ReadVariableOp0^1st_fully_connected_of_3/BiasAdd/ReadVariableOp/^1st_fully_connected_of_3/MatMul/ReadVariableOp0^1st_fully_connected_of_4/BiasAdd/ReadVariableOp/^1st_fully_connected_of_4/MatMul/ReadVariableOp ^4th_conv/BiasAdd/ReadVariableOp^4th_conv/Conv2D/ReadVariableOp ^5th_conv/BiasAdd/ReadVariableOp^5th_conv/Conv2D/ReadVariableOp ^6th_conv/BiasAdd/ReadVariableOp^6th_conv/Conv2D/ReadVariableOp6^batch_normalization_3/FusedBatchNormV3/ReadVariableOp8^batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_3/ReadVariableOp'^batch_normalization_3/ReadVariableOp_16^batch_normalization_4/FusedBatchNormV3/ReadVariableOp8^batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_4/ReadVariableOp'^batch_normalization_4/ReadVariableOp_16^batch_normalization_5/FusedBatchNormV3/ReadVariableOp8^batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_5/ReadVariableOp'^batch_normalization_5/ReadVariableOp_1 ^output_0/BiasAdd/ReadVariableOp^output_0/MatMul/ReadVariableOp ^output_1/BiasAdd/ReadVariableOp^output_1/MatMul/ReadVariableOp ^output_2/BiasAdd/ReadVariableOp^output_2/MatMul/ReadVariableOp ^output_3/BiasAdd/ReadVariableOp^output_3/MatMul/ReadVariableOp ^output_4/BiasAdd/ReadVariableOp^output_4/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*z
_input_shapesi
g:?????????dd: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2b
/1st_fully_connected_of_0/BiasAdd/ReadVariableOp/1st_fully_connected_of_0/BiasAdd/ReadVariableOp2`
.1st_fully_connected_of_0/MatMul/ReadVariableOp.1st_fully_connected_of_0/MatMul/ReadVariableOp2b
/1st_fully_connected_of_1/BiasAdd/ReadVariableOp/1st_fully_connected_of_1/BiasAdd/ReadVariableOp2`
.1st_fully_connected_of_1/MatMul/ReadVariableOp.1st_fully_connected_of_1/MatMul/ReadVariableOp2b
/1st_fully_connected_of_2/BiasAdd/ReadVariableOp/1st_fully_connected_of_2/BiasAdd/ReadVariableOp2`
.1st_fully_connected_of_2/MatMul/ReadVariableOp.1st_fully_connected_of_2/MatMul/ReadVariableOp2b
/1st_fully_connected_of_3/BiasAdd/ReadVariableOp/1st_fully_connected_of_3/BiasAdd/ReadVariableOp2`
.1st_fully_connected_of_3/MatMul/ReadVariableOp.1st_fully_connected_of_3/MatMul/ReadVariableOp2b
/1st_fully_connected_of_4/BiasAdd/ReadVariableOp/1st_fully_connected_of_4/BiasAdd/ReadVariableOp2`
.1st_fully_connected_of_4/MatMul/ReadVariableOp.1st_fully_connected_of_4/MatMul/ReadVariableOp2B
4th_conv/BiasAdd/ReadVariableOp4th_conv/BiasAdd/ReadVariableOp2@
4th_conv/Conv2D/ReadVariableOp4th_conv/Conv2D/ReadVariableOp2B
5th_conv/BiasAdd/ReadVariableOp5th_conv/BiasAdd/ReadVariableOp2@
5th_conv/Conv2D/ReadVariableOp5th_conv/Conv2D/ReadVariableOp2B
6th_conv/BiasAdd/ReadVariableOp6th_conv/BiasAdd/ReadVariableOp2@
6th_conv/Conv2D/ReadVariableOp6th_conv/Conv2D/ReadVariableOp2n
5batch_normalization_3/FusedBatchNormV3/ReadVariableOp5batch_normalization_3/FusedBatchNormV3/ReadVariableOp2r
7batch_normalization_3/FusedBatchNormV3/ReadVariableOp_17batch_normalization_3/FusedBatchNormV3/ReadVariableOp_12L
$batch_normalization_3/ReadVariableOp$batch_normalization_3/ReadVariableOp2P
&batch_normalization_3/ReadVariableOp_1&batch_normalization_3/ReadVariableOp_12n
5batch_normalization_4/FusedBatchNormV3/ReadVariableOp5batch_normalization_4/FusedBatchNormV3/ReadVariableOp2r
7batch_normalization_4/FusedBatchNormV3/ReadVariableOp_17batch_normalization_4/FusedBatchNormV3/ReadVariableOp_12L
$batch_normalization_4/ReadVariableOp$batch_normalization_4/ReadVariableOp2P
&batch_normalization_4/ReadVariableOp_1&batch_normalization_4/ReadVariableOp_12n
5batch_normalization_5/FusedBatchNormV3/ReadVariableOp5batch_normalization_5/FusedBatchNormV3/ReadVariableOp2r
7batch_normalization_5/FusedBatchNormV3/ReadVariableOp_17batch_normalization_5/FusedBatchNormV3/ReadVariableOp_12L
$batch_normalization_5/ReadVariableOp$batch_normalization_5/ReadVariableOp2P
&batch_normalization_5/ReadVariableOp_1&batch_normalization_5/ReadVariableOp_12B
output_0/BiasAdd/ReadVariableOpoutput_0/BiasAdd/ReadVariableOp2@
output_0/MatMul/ReadVariableOpoutput_0/MatMul/ReadVariableOp2B
output_1/BiasAdd/ReadVariableOpoutput_1/BiasAdd/ReadVariableOp2@
output_1/MatMul/ReadVariableOpoutput_1/MatMul/ReadVariableOp2B
output_2/BiasAdd/ReadVariableOpoutput_2/BiasAdd/ReadVariableOp2@
output_2/MatMul/ReadVariableOpoutput_2/MatMul/ReadVariableOp2B
output_3/BiasAdd/ReadVariableOpoutput_3/BiasAdd/ReadVariableOp2@
output_3/MatMul/ReadVariableOpoutput_3/MatMul/ReadVariableOp2B
output_4/BiasAdd/ReadVariableOpoutput_4/BiasAdd/ReadVariableOp2@
output_4/MatMul/ReadVariableOpoutput_4/MatMul/ReadVariableOp:W S
/
_output_shapes
:?????????dd
 
_user_specified_nameinputs
?
`
D__inference_5th_pool_layer_call_and_return_conditional_losses_107974

inputs
identity?
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4????????????????????????????????????*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
?
9__inference_1st_fully_connected_of_4_layer_call_fn_108228

inputs
unknown:	?$(
	unknown_0:(
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????(*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *]
fXRV
T__inference_1st_fully_connected_of_4_layer_call_and_return_conditional_losses_106141o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????(`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????$: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????$
 
_user_specified_nameinputs
?
?
Q__inference_batch_normalization_3_layer_call_and_return_conditional_losses_107944

inputs%
readvariableop_resource: '
readvariableop_1_resource: 6
(fusedbatchnormv3_readvariableop_resource: 8
*fusedbatchnormv3_readvariableop_1_resource: 
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype0?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+??????????????????????????? : : : : :*
epsilon%o?:*
exponential_avg_factor%
?#<?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(}
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+??????????????????????????? ?
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+??????????????????????????? : : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs
??
?"
A__inference_model_layer_call_and_return_conditional_losses_107852

inputs@
&th_conv_conv2d_readvariableop_resource: 5
'th_conv_biasadd_readvariableop_resource: ;
-batch_normalization_3_readvariableop_resource: =
/batch_normalization_3_readvariableop_1_resource: L
>batch_normalization_3_fusedbatchnormv3_readvariableop_resource: N
@batch_normalization_3_fusedbatchnormv3_readvariableop_1_resource: B
(th_conv_conv2d_readvariableop_resource_0:  7
)th_conv_biasadd_readvariableop_resource_0: ;
-batch_normalization_4_readvariableop_resource: =
/batch_normalization_4_readvariableop_1_resource: L
>batch_normalization_4_fusedbatchnormv3_readvariableop_resource: N
@batch_normalization_4_fusedbatchnormv3_readvariableop_1_resource: B
(th_conv_conv2d_readvariableop_resource_1:  7
)th_conv_biasadd_readvariableop_resource_1: ;
-batch_normalization_5_readvariableop_resource: =
/batch_normalization_5_readvariableop_1_resource: L
>batch_normalization_5_fusedbatchnormv3_readvariableop_resource: N
@batch_normalization_5_fusedbatchnormv3_readvariableop_1_resource: I
6st_fully_connected_of_4_matmul_readvariableop_resource:	?$(E
7st_fully_connected_of_4_biasadd_readvariableop_resource:(I
6st_fully_connected_of_3_matmul_readvariableop_resource:	?$(E
7st_fully_connected_of_3_biasadd_readvariableop_resource:(I
6st_fully_connected_of_2_matmul_readvariableop_resource:	?$(E
7st_fully_connected_of_2_biasadd_readvariableop_resource:(I
6st_fully_connected_of_1_matmul_readvariableop_resource:	?$(E
7st_fully_connected_of_1_biasadd_readvariableop_resource:(I
6st_fully_connected_of_0_matmul_readvariableop_resource:	?$(E
7st_fully_connected_of_0_biasadd_readvariableop_resource:(9
'output_0_matmul_readvariableop_resource:($6
(output_0_biasadd_readvariableop_resource:$9
'output_1_matmul_readvariableop_resource:($6
(output_1_biasadd_readvariableop_resource:$9
'output_2_matmul_readvariableop_resource:($6
(output_2_biasadd_readvariableop_resource:$9
'output_3_matmul_readvariableop_resource:($6
(output_3_biasadd_readvariableop_resource:$9
'output_4_matmul_readvariableop_resource:($6
(output_4_biasadd_readvariableop_resource:$
identity??/1st_fully_connected_of_0/BiasAdd/ReadVariableOp?.1st_fully_connected_of_0/MatMul/ReadVariableOp?/1st_fully_connected_of_1/BiasAdd/ReadVariableOp?.1st_fully_connected_of_1/MatMul/ReadVariableOp?/1st_fully_connected_of_2/BiasAdd/ReadVariableOp?.1st_fully_connected_of_2/MatMul/ReadVariableOp?/1st_fully_connected_of_3/BiasAdd/ReadVariableOp?.1st_fully_connected_of_3/MatMul/ReadVariableOp?/1st_fully_connected_of_4/BiasAdd/ReadVariableOp?.1st_fully_connected_of_4/MatMul/ReadVariableOp?4th_conv/BiasAdd/ReadVariableOp?4th_conv/Conv2D/ReadVariableOp?5th_conv/BiasAdd/ReadVariableOp?5th_conv/Conv2D/ReadVariableOp?6th_conv/BiasAdd/ReadVariableOp?6th_conv/Conv2D/ReadVariableOp?$batch_normalization_3/AssignNewValue?&batch_normalization_3/AssignNewValue_1?5batch_normalization_3/FusedBatchNormV3/ReadVariableOp?7batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1?$batch_normalization_3/ReadVariableOp?&batch_normalization_3/ReadVariableOp_1?$batch_normalization_4/AssignNewValue?&batch_normalization_4/AssignNewValue_1?5batch_normalization_4/FusedBatchNormV3/ReadVariableOp?7batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1?$batch_normalization_4/ReadVariableOp?&batch_normalization_4/ReadVariableOp_1?$batch_normalization_5/AssignNewValue?&batch_normalization_5/AssignNewValue_1?5batch_normalization_5/FusedBatchNormV3/ReadVariableOp?7batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1?$batch_normalization_5/ReadVariableOp?&batch_normalization_5/ReadVariableOp_1?output_0/BiasAdd/ReadVariableOp?output_0/MatMul/ReadVariableOp?output_1/BiasAdd/ReadVariableOp?output_1/MatMul/ReadVariableOp?output_2/BiasAdd/ReadVariableOp?output_2/MatMul/ReadVariableOp?output_3/BiasAdd/ReadVariableOp?output_3/MatMul/ReadVariableOp?output_4/BiasAdd/ReadVariableOp?output_4/MatMul/ReadVariableOp?
4th_conv/Conv2D/ReadVariableOpReadVariableOp&th_conv_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0?
4th_conv/Conv2DConv2Dinputs&4th_conv/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????dd *
paddingSAME*
strides
?
4th_conv/BiasAdd/ReadVariableOpReadVariableOp'th_conv_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0?
4th_conv/BiasAddBiasAdd4th_conv/Conv2D:output:0'4th_conv/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????dd j
4th_conv/ReluRelu4th_conv/BiasAdd:output:0*
T0*/
_output_shapes
:?????????dd ?
4th_pool/MaxPoolMaxPool4th_conv/Relu:activations:0*/
_output_shapes
:?????????22 *
ksize
*
paddingVALID*
strides
?
$batch_normalization_3/ReadVariableOpReadVariableOp-batch_normalization_3_readvariableop_resource*
_output_shapes
: *
dtype0?
&batch_normalization_3/ReadVariableOp_1ReadVariableOp/batch_normalization_3_readvariableop_1_resource*
_output_shapes
: *
dtype0?
5batch_normalization_3/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_3_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0?
7batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_3_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0?
&batch_normalization_3/FusedBatchNormV3FusedBatchNormV34th_pool/MaxPool:output:0,batch_normalization_3/ReadVariableOp:value:0.batch_normalization_3/ReadVariableOp_1:value:0=batch_normalization_3/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????22 : : : : :*
epsilon%o?:*
exponential_avg_factor%
?#<?
$batch_normalization_3/AssignNewValueAssignVariableOp>batch_normalization_3_fusedbatchnormv3_readvariableop_resource3batch_normalization_3/FusedBatchNormV3:batch_mean:06^batch_normalization_3/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(?
&batch_normalization_3/AssignNewValue_1AssignVariableOp@batch_normalization_3_fusedbatchnormv3_readvariableop_1_resource7batch_normalization_3/FusedBatchNormV3:batch_variance:08^batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(?
5th_conv/Conv2D/ReadVariableOpReadVariableOp(th_conv_conv2d_readvariableop_resource_0*&
_output_shapes
:  *
dtype0?
5th_conv/Conv2DConv2D*batch_normalization_3/FusedBatchNormV3:y:0&5th_conv/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????22 *
paddingSAME*
strides
?
5th_conv/BiasAdd/ReadVariableOpReadVariableOp)th_conv_biasadd_readvariableop_resource_0*
_output_shapes
: *
dtype0?
5th_conv/BiasAddBiasAdd5th_conv/Conv2D:output:0'5th_conv/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????22 j
5th_conv/ReluRelu5th_conv/BiasAdd:output:0*
T0*/
_output_shapes
:?????????22 ?
5th_pool/MaxPoolMaxPool5th_conv/Relu:activations:0*/
_output_shapes
:????????? *
ksize
*
paddingVALID*
strides
?
$batch_normalization_4/ReadVariableOpReadVariableOp-batch_normalization_4_readvariableop_resource*
_output_shapes
: *
dtype0?
&batch_normalization_4/ReadVariableOp_1ReadVariableOp/batch_normalization_4_readvariableop_1_resource*
_output_shapes
: *
dtype0?
5batch_normalization_4/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_4_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0?
7batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_4_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0?
&batch_normalization_4/FusedBatchNormV3FusedBatchNormV35th_pool/MaxPool:output:0,batch_normalization_4/ReadVariableOp:value:0.batch_normalization_4/ReadVariableOp_1:value:0=batch_normalization_4/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:????????? : : : : :*
epsilon%o?:*
exponential_avg_factor%
?#<?
$batch_normalization_4/AssignNewValueAssignVariableOp>batch_normalization_4_fusedbatchnormv3_readvariableop_resource3batch_normalization_4/FusedBatchNormV3:batch_mean:06^batch_normalization_4/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(?
&batch_normalization_4/AssignNewValue_1AssignVariableOp@batch_normalization_4_fusedbatchnormv3_readvariableop_1_resource7batch_normalization_4/FusedBatchNormV3:batch_variance:08^batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(?
6th_conv/Conv2D/ReadVariableOpReadVariableOp(th_conv_conv2d_readvariableop_resource_1*&
_output_shapes
:  *
dtype0?
6th_conv/Conv2DConv2D*batch_normalization_4/FusedBatchNormV3:y:0&6th_conv/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? *
paddingSAME*
strides
?
6th_conv/BiasAdd/ReadVariableOpReadVariableOp)th_conv_biasadd_readvariableop_resource_1*
_output_shapes
: *
dtype0?
6th_conv/BiasAddBiasAdd6th_conv/Conv2D:output:0'6th_conv/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? j
6th_conv/ReluRelu6th_conv/BiasAdd:output:0*
T0*/
_output_shapes
:????????? ?
6th_pool/MaxPoolMaxPool6th_conv/Relu:activations:0*/
_output_shapes
:????????? *
ksize
*
paddingVALID*
strides
?
$batch_normalization_5/ReadVariableOpReadVariableOp-batch_normalization_5_readvariableop_resource*
_output_shapes
: *
dtype0?
&batch_normalization_5/ReadVariableOp_1ReadVariableOp/batch_normalization_5_readvariableop_1_resource*
_output_shapes
: *
dtype0?
5batch_normalization_5/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_5_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0?
7batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_5_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0?
&batch_normalization_5/FusedBatchNormV3FusedBatchNormV36th_pool/MaxPool:output:0,batch_normalization_5/ReadVariableOp:value:0.batch_normalization_5/ReadVariableOp_1:value:0=batch_normalization_5/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:????????? : : : : :*
epsilon%o?:*
exponential_avg_factor%
?#<?
$batch_normalization_5/AssignNewValueAssignVariableOp>batch_normalization_5_fusedbatchnormv3_readvariableop_resource3batch_normalization_5/FusedBatchNormV3:batch_mean:06^batch_normalization_5/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(?
&batch_normalization_5/AssignNewValue_1AssignVariableOp@batch_normalization_5_fusedbatchnormv3_readvariableop_1_resource7batch_normalization_5/FusedBatchNormV3:batch_variance:08^batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape([

flat/ConstConst*
_output_shapes
:*
dtype0*
valueB"????   ?
flat/ReshapeReshape*batch_normalization_5/FusedBatchNormV3:y:0flat/Const:output:0*
T0*(
_output_shapes
:??????????$?
.1st_fully_connected_of_4/MatMul/ReadVariableOpReadVariableOp6st_fully_connected_of_4_matmul_readvariableop_resource*
_output_shapes
:	?$(*
dtype0?
1st_fully_connected_of_4/MatMulMatMulflat/Reshape:output:061st_fully_connected_of_4/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????(?
/1st_fully_connected_of_4/BiasAdd/ReadVariableOpReadVariableOp7st_fully_connected_of_4_biasadd_readvariableop_resource*
_output_shapes
:(*
dtype0?
 1st_fully_connected_of_4/BiasAddBiasAdd)1st_fully_connected_of_4/MatMul:product:071st_fully_connected_of_4/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????(?
1st_fully_connected_of_4/ReluRelu)1st_fully_connected_of_4/BiasAdd:output:0*
T0*'
_output_shapes
:?????????(?
.1st_fully_connected_of_3/MatMul/ReadVariableOpReadVariableOp6st_fully_connected_of_3_matmul_readvariableop_resource*
_output_shapes
:	?$(*
dtype0?
1st_fully_connected_of_3/MatMulMatMulflat/Reshape:output:061st_fully_connected_of_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????(?
/1st_fully_connected_of_3/BiasAdd/ReadVariableOpReadVariableOp7st_fully_connected_of_3_biasadd_readvariableop_resource*
_output_shapes
:(*
dtype0?
 1st_fully_connected_of_3/BiasAddBiasAdd)1st_fully_connected_of_3/MatMul:product:071st_fully_connected_of_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????(?
1st_fully_connected_of_3/ReluRelu)1st_fully_connected_of_3/BiasAdd:output:0*
T0*'
_output_shapes
:?????????(?
.1st_fully_connected_of_2/MatMul/ReadVariableOpReadVariableOp6st_fully_connected_of_2_matmul_readvariableop_resource*
_output_shapes
:	?$(*
dtype0?
1st_fully_connected_of_2/MatMulMatMulflat/Reshape:output:061st_fully_connected_of_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????(?
/1st_fully_connected_of_2/BiasAdd/ReadVariableOpReadVariableOp7st_fully_connected_of_2_biasadd_readvariableop_resource*
_output_shapes
:(*
dtype0?
 1st_fully_connected_of_2/BiasAddBiasAdd)1st_fully_connected_of_2/MatMul:product:071st_fully_connected_of_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????(?
1st_fully_connected_of_2/ReluRelu)1st_fully_connected_of_2/BiasAdd:output:0*
T0*'
_output_shapes
:?????????(?
.1st_fully_connected_of_1/MatMul/ReadVariableOpReadVariableOp6st_fully_connected_of_1_matmul_readvariableop_resource*
_output_shapes
:	?$(*
dtype0?
1st_fully_connected_of_1/MatMulMatMulflat/Reshape:output:061st_fully_connected_of_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????(?
/1st_fully_connected_of_1/BiasAdd/ReadVariableOpReadVariableOp7st_fully_connected_of_1_biasadd_readvariableop_resource*
_output_shapes
:(*
dtype0?
 1st_fully_connected_of_1/BiasAddBiasAdd)1st_fully_connected_of_1/MatMul:product:071st_fully_connected_of_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????(?
1st_fully_connected_of_1/ReluRelu)1st_fully_connected_of_1/BiasAdd:output:0*
T0*'
_output_shapes
:?????????(?
.1st_fully_connected_of_0/MatMul/ReadVariableOpReadVariableOp6st_fully_connected_of_0_matmul_readvariableop_resource*
_output_shapes
:	?$(*
dtype0?
1st_fully_connected_of_0/MatMulMatMulflat/Reshape:output:061st_fully_connected_of_0/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????(?
/1st_fully_connected_of_0/BiasAdd/ReadVariableOpReadVariableOp7st_fully_connected_of_0_biasadd_readvariableop_resource*
_output_shapes
:(*
dtype0?
 1st_fully_connected_of_0/BiasAddBiasAdd)1st_fully_connected_of_0/MatMul:product:071st_fully_connected_of_0/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????(?
1st_fully_connected_of_0/ReluRelu)1st_fully_connected_of_0/BiasAdd:output:0*
T0*'
_output_shapes
:?????????(\
dropout_4/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
dropout_4/dropout/MulMul+1st_fully_connected_of_4/Relu:activations:0 dropout_4/dropout/Const:output:0*
T0*'
_output_shapes
:?????????(r
dropout_4/dropout/ShapeShape+1st_fully_connected_of_4/Relu:activations:0*
T0*
_output_shapes
:?
.dropout_4/dropout/random_uniform/RandomUniformRandomUniform dropout_4/dropout/Shape:output:0*
T0*'
_output_shapes
:?????????(*
dtype0e
 dropout_4/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>?
dropout_4/dropout/GreaterEqualGreaterEqual7dropout_4/dropout/random_uniform/RandomUniform:output:0)dropout_4/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????(?
dropout_4/dropout/CastCast"dropout_4/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????(?
dropout_4/dropout/Mul_1Muldropout_4/dropout/Mul:z:0dropout_4/dropout/Cast:y:0*
T0*'
_output_shapes
:?????????(\
dropout_3/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
dropout_3/dropout/MulMul+1st_fully_connected_of_3/Relu:activations:0 dropout_3/dropout/Const:output:0*
T0*'
_output_shapes
:?????????(r
dropout_3/dropout/ShapeShape+1st_fully_connected_of_3/Relu:activations:0*
T0*
_output_shapes
:?
.dropout_3/dropout/random_uniform/RandomUniformRandomUniform dropout_3/dropout/Shape:output:0*
T0*'
_output_shapes
:?????????(*
dtype0e
 dropout_3/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>?
dropout_3/dropout/GreaterEqualGreaterEqual7dropout_3/dropout/random_uniform/RandomUniform:output:0)dropout_3/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????(?
dropout_3/dropout/CastCast"dropout_3/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????(?
dropout_3/dropout/Mul_1Muldropout_3/dropout/Mul:z:0dropout_3/dropout/Cast:y:0*
T0*'
_output_shapes
:?????????(\
dropout_2/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
dropout_2/dropout/MulMul+1st_fully_connected_of_2/Relu:activations:0 dropout_2/dropout/Const:output:0*
T0*'
_output_shapes
:?????????(r
dropout_2/dropout/ShapeShape+1st_fully_connected_of_2/Relu:activations:0*
T0*
_output_shapes
:?
.dropout_2/dropout/random_uniform/RandomUniformRandomUniform dropout_2/dropout/Shape:output:0*
T0*'
_output_shapes
:?????????(*
dtype0e
 dropout_2/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>?
dropout_2/dropout/GreaterEqualGreaterEqual7dropout_2/dropout/random_uniform/RandomUniform:output:0)dropout_2/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????(?
dropout_2/dropout/CastCast"dropout_2/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????(?
dropout_2/dropout/Mul_1Muldropout_2/dropout/Mul:z:0dropout_2/dropout/Cast:y:0*
T0*'
_output_shapes
:?????????(\
dropout_1/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
dropout_1/dropout/MulMul+1st_fully_connected_of_1/Relu:activations:0 dropout_1/dropout/Const:output:0*
T0*'
_output_shapes
:?????????(r
dropout_1/dropout/ShapeShape+1st_fully_connected_of_1/Relu:activations:0*
T0*
_output_shapes
:?
.dropout_1/dropout/random_uniform/RandomUniformRandomUniform dropout_1/dropout/Shape:output:0*
T0*'
_output_shapes
:?????????(*
dtype0e
 dropout_1/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>?
dropout_1/dropout/GreaterEqualGreaterEqual7dropout_1/dropout/random_uniform/RandomUniform:output:0)dropout_1/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????(?
dropout_1/dropout/CastCast"dropout_1/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????(?
dropout_1/dropout/Mul_1Muldropout_1/dropout/Mul:z:0dropout_1/dropout/Cast:y:0*
T0*'
_output_shapes
:?????????(Z
dropout/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
dropout/dropout/MulMul+1st_fully_connected_of_0/Relu:activations:0dropout/dropout/Const:output:0*
T0*'
_output_shapes
:?????????(p
dropout/dropout/ShapeShape+1st_fully_connected_of_0/Relu:activations:0*
T0*
_output_shapes
:?
,dropout/dropout/random_uniform/RandomUniformRandomUniformdropout/dropout/Shape:output:0*
T0*'
_output_shapes
:?????????(*
dtype0c
dropout/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>?
dropout/dropout/GreaterEqualGreaterEqual5dropout/dropout/random_uniform/RandomUniform:output:0'dropout/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????(
dropout/dropout/CastCast dropout/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????(?
dropout/dropout/Mul_1Muldropout/dropout/Mul:z:0dropout/dropout/Cast:y:0*
T0*'
_output_shapes
:?????????(?
output_0/MatMul/ReadVariableOpReadVariableOp'output_0_matmul_readvariableop_resource*
_output_shapes

:($*
dtype0?
output_0/MatMulMatMuldropout/dropout/Mul_1:z:0&output_0/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????$?
output_0/BiasAdd/ReadVariableOpReadVariableOp(output_0_biasadd_readvariableop_resource*
_output_shapes
:$*
dtype0?
output_0/BiasAddBiasAddoutput_0/MatMul:product:0'output_0/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????$h
output_0/SoftmaxSoftmaxoutput_0/BiasAdd:output:0*
T0*'
_output_shapes
:?????????$?
output_1/MatMul/ReadVariableOpReadVariableOp'output_1_matmul_readvariableop_resource*
_output_shapes

:($*
dtype0?
output_1/MatMulMatMuldropout_1/dropout/Mul_1:z:0&output_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????$?
output_1/BiasAdd/ReadVariableOpReadVariableOp(output_1_biasadd_readvariableop_resource*
_output_shapes
:$*
dtype0?
output_1/BiasAddBiasAddoutput_1/MatMul:product:0'output_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????$h
output_1/SoftmaxSoftmaxoutput_1/BiasAdd:output:0*
T0*'
_output_shapes
:?????????$?
output_2/MatMul/ReadVariableOpReadVariableOp'output_2_matmul_readvariableop_resource*
_output_shapes

:($*
dtype0?
output_2/MatMulMatMuldropout_2/dropout/Mul_1:z:0&output_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????$?
output_2/BiasAdd/ReadVariableOpReadVariableOp(output_2_biasadd_readvariableop_resource*
_output_shapes
:$*
dtype0?
output_2/BiasAddBiasAddoutput_2/MatMul:product:0'output_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????$h
output_2/SoftmaxSoftmaxoutput_2/BiasAdd:output:0*
T0*'
_output_shapes
:?????????$?
output_3/MatMul/ReadVariableOpReadVariableOp'output_3_matmul_readvariableop_resource*
_output_shapes

:($*
dtype0?
output_3/MatMulMatMuldropout_3/dropout/Mul_1:z:0&output_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????$?
output_3/BiasAdd/ReadVariableOpReadVariableOp(output_3_biasadd_readvariableop_resource*
_output_shapes
:$*
dtype0?
output_3/BiasAddBiasAddoutput_3/MatMul:product:0'output_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????$h
output_3/SoftmaxSoftmaxoutput_3/BiasAdd:output:0*
T0*'
_output_shapes
:?????????$?
output_4/MatMul/ReadVariableOpReadVariableOp'output_4_matmul_readvariableop_resource*
_output_shapes

:($*
dtype0?
output_4/MatMulMatMuldropout_4/dropout/Mul_1:z:0&output_4/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????$?
output_4/BiasAdd/ReadVariableOpReadVariableOp(output_4_biasadd_readvariableop_resource*
_output_shapes
:$*
dtype0?
output_4/BiasAddBiasAddoutput_4/MatMul:product:0'output_4/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????$h
output_4/SoftmaxSoftmaxoutput_4/BiasAdd:output:0*
T0*'
_output_shapes
:?????????$?
tf.convert_to_tensor_1/CastCastoutput_0/Softmax:softmax:0*

DstT0*

SrcT0*'
_output_shapes
:?????????$?
tf.convert_to_tensor_1/Cast_1Castoutput_1/Softmax:softmax:0*

DstT0*

SrcT0*'
_output_shapes
:?????????$?
tf.convert_to_tensor_1/Cast_2Castoutput_2/Softmax:softmax:0*

DstT0*

SrcT0*'
_output_shapes
:?????????$?
tf.convert_to_tensor_1/Cast_3Castoutput_3/Softmax:softmax:0*

DstT0*

SrcT0*'
_output_shapes
:?????????$?
tf.convert_to_tensor_1/Cast_4Castoutput_4/Softmax:softmax:0*

DstT0*

SrcT0*'
_output_shapes
:?????????$?
tf.convert_to_tensor_1/packedPacktf.convert_to_tensor_1/Cast:y:0!tf.convert_to_tensor_1/Cast_1:y:0!tf.convert_to_tensor_1/Cast_2:y:0!tf.convert_to_tensor_1/Cast_3:y:0!tf.convert_to_tensor_1/Cast_4:y:0*
N*
T0*+
_output_shapes
:?????????$|
'tf.compat.v1.transpose_1/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          ?
"tf.compat.v1.transpose_1/transpose	Transpose&tf.convert_to_tensor_1/packed:output:00tf.compat.v1.transpose_1/transpose/perm:output:0*
T0*+
_output_shapes
:?????????$y
IdentityIdentity&tf.compat.v1.transpose_1/transpose:y:0^NoOp*
T0*+
_output_shapes
:?????????$?
NoOpNoOp0^1st_fully_connected_of_0/BiasAdd/ReadVariableOp/^1st_fully_connected_of_0/MatMul/ReadVariableOp0^1st_fully_connected_of_1/BiasAdd/ReadVariableOp/^1st_fully_connected_of_1/MatMul/ReadVariableOp0^1st_fully_connected_of_2/BiasAdd/ReadVariableOp/^1st_fully_connected_of_2/MatMul/ReadVariableOp0^1st_fully_connected_of_3/BiasAdd/ReadVariableOp/^1st_fully_connected_of_3/MatMul/ReadVariableOp0^1st_fully_connected_of_4/BiasAdd/ReadVariableOp/^1st_fully_connected_of_4/MatMul/ReadVariableOp ^4th_conv/BiasAdd/ReadVariableOp^4th_conv/Conv2D/ReadVariableOp ^5th_conv/BiasAdd/ReadVariableOp^5th_conv/Conv2D/ReadVariableOp ^6th_conv/BiasAdd/ReadVariableOp^6th_conv/Conv2D/ReadVariableOp%^batch_normalization_3/AssignNewValue'^batch_normalization_3/AssignNewValue_16^batch_normalization_3/FusedBatchNormV3/ReadVariableOp8^batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_3/ReadVariableOp'^batch_normalization_3/ReadVariableOp_1%^batch_normalization_4/AssignNewValue'^batch_normalization_4/AssignNewValue_16^batch_normalization_4/FusedBatchNormV3/ReadVariableOp8^batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_4/ReadVariableOp'^batch_normalization_4/ReadVariableOp_1%^batch_normalization_5/AssignNewValue'^batch_normalization_5/AssignNewValue_16^batch_normalization_5/FusedBatchNormV3/ReadVariableOp8^batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_5/ReadVariableOp'^batch_normalization_5/ReadVariableOp_1 ^output_0/BiasAdd/ReadVariableOp^output_0/MatMul/ReadVariableOp ^output_1/BiasAdd/ReadVariableOp^output_1/MatMul/ReadVariableOp ^output_2/BiasAdd/ReadVariableOp^output_2/MatMul/ReadVariableOp ^output_3/BiasAdd/ReadVariableOp^output_3/MatMul/ReadVariableOp ^output_4/BiasAdd/ReadVariableOp^output_4/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*z
_input_shapesi
g:?????????dd: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2b
/1st_fully_connected_of_0/BiasAdd/ReadVariableOp/1st_fully_connected_of_0/BiasAdd/ReadVariableOp2`
.1st_fully_connected_of_0/MatMul/ReadVariableOp.1st_fully_connected_of_0/MatMul/ReadVariableOp2b
/1st_fully_connected_of_1/BiasAdd/ReadVariableOp/1st_fully_connected_of_1/BiasAdd/ReadVariableOp2`
.1st_fully_connected_of_1/MatMul/ReadVariableOp.1st_fully_connected_of_1/MatMul/ReadVariableOp2b
/1st_fully_connected_of_2/BiasAdd/ReadVariableOp/1st_fully_connected_of_2/BiasAdd/ReadVariableOp2`
.1st_fully_connected_of_2/MatMul/ReadVariableOp.1st_fully_connected_of_2/MatMul/ReadVariableOp2b
/1st_fully_connected_of_3/BiasAdd/ReadVariableOp/1st_fully_connected_of_3/BiasAdd/ReadVariableOp2`
.1st_fully_connected_of_3/MatMul/ReadVariableOp.1st_fully_connected_of_3/MatMul/ReadVariableOp2b
/1st_fully_connected_of_4/BiasAdd/ReadVariableOp/1st_fully_connected_of_4/BiasAdd/ReadVariableOp2`
.1st_fully_connected_of_4/MatMul/ReadVariableOp.1st_fully_connected_of_4/MatMul/ReadVariableOp2B
4th_conv/BiasAdd/ReadVariableOp4th_conv/BiasAdd/ReadVariableOp2@
4th_conv/Conv2D/ReadVariableOp4th_conv/Conv2D/ReadVariableOp2B
5th_conv/BiasAdd/ReadVariableOp5th_conv/BiasAdd/ReadVariableOp2@
5th_conv/Conv2D/ReadVariableOp5th_conv/Conv2D/ReadVariableOp2B
6th_conv/BiasAdd/ReadVariableOp6th_conv/BiasAdd/ReadVariableOp2@
6th_conv/Conv2D/ReadVariableOp6th_conv/Conv2D/ReadVariableOp2L
$batch_normalization_3/AssignNewValue$batch_normalization_3/AssignNewValue2P
&batch_normalization_3/AssignNewValue_1&batch_normalization_3/AssignNewValue_12n
5batch_normalization_3/FusedBatchNormV3/ReadVariableOp5batch_normalization_3/FusedBatchNormV3/ReadVariableOp2r
7batch_normalization_3/FusedBatchNormV3/ReadVariableOp_17batch_normalization_3/FusedBatchNormV3/ReadVariableOp_12L
$batch_normalization_3/ReadVariableOp$batch_normalization_3/ReadVariableOp2P
&batch_normalization_3/ReadVariableOp_1&batch_normalization_3/ReadVariableOp_12L
$batch_normalization_4/AssignNewValue$batch_normalization_4/AssignNewValue2P
&batch_normalization_4/AssignNewValue_1&batch_normalization_4/AssignNewValue_12n
5batch_normalization_4/FusedBatchNormV3/ReadVariableOp5batch_normalization_4/FusedBatchNormV3/ReadVariableOp2r
7batch_normalization_4/FusedBatchNormV3/ReadVariableOp_17batch_normalization_4/FusedBatchNormV3/ReadVariableOp_12L
$batch_normalization_4/ReadVariableOp$batch_normalization_4/ReadVariableOp2P
&batch_normalization_4/ReadVariableOp_1&batch_normalization_4/ReadVariableOp_12L
$batch_normalization_5/AssignNewValue$batch_normalization_5/AssignNewValue2P
&batch_normalization_5/AssignNewValue_1&batch_normalization_5/AssignNewValue_12n
5batch_normalization_5/FusedBatchNormV3/ReadVariableOp5batch_normalization_5/FusedBatchNormV3/ReadVariableOp2r
7batch_normalization_5/FusedBatchNormV3/ReadVariableOp_17batch_normalization_5/FusedBatchNormV3/ReadVariableOp_12L
$batch_normalization_5/ReadVariableOp$batch_normalization_5/ReadVariableOp2P
&batch_normalization_5/ReadVariableOp_1&batch_normalization_5/ReadVariableOp_12B
output_0/BiasAdd/ReadVariableOpoutput_0/BiasAdd/ReadVariableOp2@
output_0/MatMul/ReadVariableOpoutput_0/MatMul/ReadVariableOp2B
output_1/BiasAdd/ReadVariableOpoutput_1/BiasAdd/ReadVariableOp2@
output_1/MatMul/ReadVariableOpoutput_1/MatMul/ReadVariableOp2B
output_2/BiasAdd/ReadVariableOpoutput_2/BiasAdd/ReadVariableOp2@
output_2/MatMul/ReadVariableOpoutput_2/MatMul/ReadVariableOp2B
output_3/BiasAdd/ReadVariableOpoutput_3/BiasAdd/ReadVariableOp2@
output_3/MatMul/ReadVariableOpoutput_3/MatMul/ReadVariableOp2B
output_4/BiasAdd/ReadVariableOpoutput_4/BiasAdd/ReadVariableOp2@
output_4/MatMul/ReadVariableOpoutput_4/MatMul/ReadVariableOp:W S
/
_output_shapes
:?????????dd
 
_user_specified_nameinputs
?	
?
6__inference_batch_normalization_5_layer_call_fn_108092

inputs
unknown: 
	unknown_0: 
	unknown_1: 
	unknown_2: 
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+??????????????????????????? *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Z
fURS
Q__inference_batch_normalization_5_layer_call_and_return_conditional_losses_106023?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+??????????????????????????? `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+??????????????????????????? : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs
?	
d
E__inference_dropout_2_layer_call_and_return_conditional_losses_106539

inputs
identity?R
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??d
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:?????????(C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:?????????(*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????(o
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????(i
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:?????????(Y
IdentityIdentitydropout/Mul_1:z:0*
T0*'
_output_shapes
:?????????("
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:?????????(:O K
'
_output_shapes
:?????????(
 
_user_specified_nameinputs
??
?3
__inference__traced_save_108830
file_prefix.
*savev2_4th_conv_kernel_read_readvariableop,
(savev2_4th_conv_bias_read_readvariableop:
6savev2_batch_normalization_3_gamma_read_readvariableop9
5savev2_batch_normalization_3_beta_read_readvariableop@
<savev2_batch_normalization_3_moving_mean_read_readvariableopD
@savev2_batch_normalization_3_moving_variance_read_readvariableop.
*savev2_5th_conv_kernel_read_readvariableop,
(savev2_5th_conv_bias_read_readvariableop:
6savev2_batch_normalization_4_gamma_read_readvariableop9
5savev2_batch_normalization_4_beta_read_readvariableop@
<savev2_batch_normalization_4_moving_mean_read_readvariableopD
@savev2_batch_normalization_4_moving_variance_read_readvariableop.
*savev2_6th_conv_kernel_read_readvariableop,
(savev2_6th_conv_bias_read_readvariableop:
6savev2_batch_normalization_5_gamma_read_readvariableop9
5savev2_batch_normalization_5_beta_read_readvariableop@
<savev2_batch_normalization_5_moving_mean_read_readvariableopD
@savev2_batch_normalization_5_moving_variance_read_readvariableop>
:savev2_1st_fully_connected_of_0_kernel_read_readvariableop<
8savev2_1st_fully_connected_of_0_bias_read_readvariableop>
:savev2_1st_fully_connected_of_1_kernel_read_readvariableop<
8savev2_1st_fully_connected_of_1_bias_read_readvariableop>
:savev2_1st_fully_connected_of_2_kernel_read_readvariableop<
8savev2_1st_fully_connected_of_2_bias_read_readvariableop>
:savev2_1st_fully_connected_of_3_kernel_read_readvariableop<
8savev2_1st_fully_connected_of_3_bias_read_readvariableop>
:savev2_1st_fully_connected_of_4_kernel_read_readvariableop<
8savev2_1st_fully_connected_of_4_bias_read_readvariableop.
*savev2_output_0_kernel_read_readvariableop,
(savev2_output_0_bias_read_readvariableop.
*savev2_output_1_kernel_read_readvariableop,
(savev2_output_1_bias_read_readvariableop.
*savev2_output_2_kernel_read_readvariableop,
(savev2_output_2_bias_read_readvariableop.
*savev2_output_3_kernel_read_readvariableop,
(savev2_output_3_bias_read_readvariableop.
*savev2_output_4_kernel_read_readvariableop,
(savev2_output_4_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop5
1savev2_adam_4th_conv_kernel_m_read_readvariableop3
/savev2_adam_4th_conv_bias_m_read_readvariableopA
=savev2_adam_batch_normalization_3_gamma_m_read_readvariableop@
<savev2_adam_batch_normalization_3_beta_m_read_readvariableop5
1savev2_adam_5th_conv_kernel_m_read_readvariableop3
/savev2_adam_5th_conv_bias_m_read_readvariableopA
=savev2_adam_batch_normalization_4_gamma_m_read_readvariableop@
<savev2_adam_batch_normalization_4_beta_m_read_readvariableop5
1savev2_adam_6th_conv_kernel_m_read_readvariableop3
/savev2_adam_6th_conv_bias_m_read_readvariableopA
=savev2_adam_batch_normalization_5_gamma_m_read_readvariableop@
<savev2_adam_batch_normalization_5_beta_m_read_readvariableopE
Asavev2_adam_1st_fully_connected_of_0_kernel_m_read_readvariableopC
?savev2_adam_1st_fully_connected_of_0_bias_m_read_readvariableopE
Asavev2_adam_1st_fully_connected_of_1_kernel_m_read_readvariableopC
?savev2_adam_1st_fully_connected_of_1_bias_m_read_readvariableopE
Asavev2_adam_1st_fully_connected_of_2_kernel_m_read_readvariableopC
?savev2_adam_1st_fully_connected_of_2_bias_m_read_readvariableopE
Asavev2_adam_1st_fully_connected_of_3_kernel_m_read_readvariableopC
?savev2_adam_1st_fully_connected_of_3_bias_m_read_readvariableopE
Asavev2_adam_1st_fully_connected_of_4_kernel_m_read_readvariableopC
?savev2_adam_1st_fully_connected_of_4_bias_m_read_readvariableop5
1savev2_adam_output_0_kernel_m_read_readvariableop3
/savev2_adam_output_0_bias_m_read_readvariableop5
1savev2_adam_output_1_kernel_m_read_readvariableop3
/savev2_adam_output_1_bias_m_read_readvariableop5
1savev2_adam_output_2_kernel_m_read_readvariableop3
/savev2_adam_output_2_bias_m_read_readvariableop5
1savev2_adam_output_3_kernel_m_read_readvariableop3
/savev2_adam_output_3_bias_m_read_readvariableop5
1savev2_adam_output_4_kernel_m_read_readvariableop3
/savev2_adam_output_4_bias_m_read_readvariableop5
1savev2_adam_4th_conv_kernel_v_read_readvariableop3
/savev2_adam_4th_conv_bias_v_read_readvariableopA
=savev2_adam_batch_normalization_3_gamma_v_read_readvariableop@
<savev2_adam_batch_normalization_3_beta_v_read_readvariableop5
1savev2_adam_5th_conv_kernel_v_read_readvariableop3
/savev2_adam_5th_conv_bias_v_read_readvariableopA
=savev2_adam_batch_normalization_4_gamma_v_read_readvariableop@
<savev2_adam_batch_normalization_4_beta_v_read_readvariableop5
1savev2_adam_6th_conv_kernel_v_read_readvariableop3
/savev2_adam_6th_conv_bias_v_read_readvariableopA
=savev2_adam_batch_normalization_5_gamma_v_read_readvariableop@
<savev2_adam_batch_normalization_5_beta_v_read_readvariableopE
Asavev2_adam_1st_fully_connected_of_0_kernel_v_read_readvariableopC
?savev2_adam_1st_fully_connected_of_0_bias_v_read_readvariableopE
Asavev2_adam_1st_fully_connected_of_1_kernel_v_read_readvariableopC
?savev2_adam_1st_fully_connected_of_1_bias_v_read_readvariableopE
Asavev2_adam_1st_fully_connected_of_2_kernel_v_read_readvariableopC
?savev2_adam_1st_fully_connected_of_2_bias_v_read_readvariableopE
Asavev2_adam_1st_fully_connected_of_3_kernel_v_read_readvariableopC
?savev2_adam_1st_fully_connected_of_3_bias_v_read_readvariableopE
Asavev2_adam_1st_fully_connected_of_4_kernel_v_read_readvariableopC
?savev2_adam_1st_fully_connected_of_4_bias_v_read_readvariableop5
1savev2_adam_output_0_kernel_v_read_readvariableop3
/savev2_adam_output_0_bias_v_read_readvariableop5
1savev2_adam_output_1_kernel_v_read_readvariableop3
/savev2_adam_output_1_bias_v_read_readvariableop5
1savev2_adam_output_2_kernel_v_read_readvariableop3
/savev2_adam_output_2_bias_v_read_readvariableop5
1savev2_adam_output_3_kernel_v_read_readvariableop3
/savev2_adam_output_3_bias_v_read_readvariableop5
1savev2_adam_output_4_kernel_v_read_readvariableop3
/savev2_adam_output_4_bias_v_read_readvariableop
savev2_const

identity_1??MergeV2Checkpointsw
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*Z
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.parta
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part?
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: f

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: L

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :f
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : ?
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: ?>
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:p*
dtype0*?>
value?>B?>pB6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-1/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-1/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-3/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-3/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-5/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-5/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-5/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-9/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-9/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-10/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-10/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-11/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-11/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-12/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-12/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-13/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-13/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-14/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-14/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-15/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-15/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-5/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-10/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-10/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-11/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-11/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-12/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-12/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-13/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-13/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-14/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-14/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-15/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-15/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-5/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-10/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-10/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-11/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-11/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-12/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-12/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-13/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-13/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-14/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-14/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-15/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-15/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH?
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:p*
dtype0*?
value?B?pB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B ?1
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0*savev2_4th_conv_kernel_read_readvariableop(savev2_4th_conv_bias_read_readvariableop6savev2_batch_normalization_3_gamma_read_readvariableop5savev2_batch_normalization_3_beta_read_readvariableop<savev2_batch_normalization_3_moving_mean_read_readvariableop@savev2_batch_normalization_3_moving_variance_read_readvariableop*savev2_5th_conv_kernel_read_readvariableop(savev2_5th_conv_bias_read_readvariableop6savev2_batch_normalization_4_gamma_read_readvariableop5savev2_batch_normalization_4_beta_read_readvariableop<savev2_batch_normalization_4_moving_mean_read_readvariableop@savev2_batch_normalization_4_moving_variance_read_readvariableop*savev2_6th_conv_kernel_read_readvariableop(savev2_6th_conv_bias_read_readvariableop6savev2_batch_normalization_5_gamma_read_readvariableop5savev2_batch_normalization_5_beta_read_readvariableop<savev2_batch_normalization_5_moving_mean_read_readvariableop@savev2_batch_normalization_5_moving_variance_read_readvariableop:savev2_1st_fully_connected_of_0_kernel_read_readvariableop8savev2_1st_fully_connected_of_0_bias_read_readvariableop:savev2_1st_fully_connected_of_1_kernel_read_readvariableop8savev2_1st_fully_connected_of_1_bias_read_readvariableop:savev2_1st_fully_connected_of_2_kernel_read_readvariableop8savev2_1st_fully_connected_of_2_bias_read_readvariableop:savev2_1st_fully_connected_of_3_kernel_read_readvariableop8savev2_1st_fully_connected_of_3_bias_read_readvariableop:savev2_1st_fully_connected_of_4_kernel_read_readvariableop8savev2_1st_fully_connected_of_4_bias_read_readvariableop*savev2_output_0_kernel_read_readvariableop(savev2_output_0_bias_read_readvariableop*savev2_output_1_kernel_read_readvariableop(savev2_output_1_bias_read_readvariableop*savev2_output_2_kernel_read_readvariableop(savev2_output_2_bias_read_readvariableop*savev2_output_3_kernel_read_readvariableop(savev2_output_3_bias_read_readvariableop*savev2_output_4_kernel_read_readvariableop(savev2_output_4_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop1savev2_adam_4th_conv_kernel_m_read_readvariableop/savev2_adam_4th_conv_bias_m_read_readvariableop=savev2_adam_batch_normalization_3_gamma_m_read_readvariableop<savev2_adam_batch_normalization_3_beta_m_read_readvariableop1savev2_adam_5th_conv_kernel_m_read_readvariableop/savev2_adam_5th_conv_bias_m_read_readvariableop=savev2_adam_batch_normalization_4_gamma_m_read_readvariableop<savev2_adam_batch_normalization_4_beta_m_read_readvariableop1savev2_adam_6th_conv_kernel_m_read_readvariableop/savev2_adam_6th_conv_bias_m_read_readvariableop=savev2_adam_batch_normalization_5_gamma_m_read_readvariableop<savev2_adam_batch_normalization_5_beta_m_read_readvariableopAsavev2_adam_1st_fully_connected_of_0_kernel_m_read_readvariableop?savev2_adam_1st_fully_connected_of_0_bias_m_read_readvariableopAsavev2_adam_1st_fully_connected_of_1_kernel_m_read_readvariableop?savev2_adam_1st_fully_connected_of_1_bias_m_read_readvariableopAsavev2_adam_1st_fully_connected_of_2_kernel_m_read_readvariableop?savev2_adam_1st_fully_connected_of_2_bias_m_read_readvariableopAsavev2_adam_1st_fully_connected_of_3_kernel_m_read_readvariableop?savev2_adam_1st_fully_connected_of_3_bias_m_read_readvariableopAsavev2_adam_1st_fully_connected_of_4_kernel_m_read_readvariableop?savev2_adam_1st_fully_connected_of_4_bias_m_read_readvariableop1savev2_adam_output_0_kernel_m_read_readvariableop/savev2_adam_output_0_bias_m_read_readvariableop1savev2_adam_output_1_kernel_m_read_readvariableop/savev2_adam_output_1_bias_m_read_readvariableop1savev2_adam_output_2_kernel_m_read_readvariableop/savev2_adam_output_2_bias_m_read_readvariableop1savev2_adam_output_3_kernel_m_read_readvariableop/savev2_adam_output_3_bias_m_read_readvariableop1savev2_adam_output_4_kernel_m_read_readvariableop/savev2_adam_output_4_bias_m_read_readvariableop1savev2_adam_4th_conv_kernel_v_read_readvariableop/savev2_adam_4th_conv_bias_v_read_readvariableop=savev2_adam_batch_normalization_3_gamma_v_read_readvariableop<savev2_adam_batch_normalization_3_beta_v_read_readvariableop1savev2_adam_5th_conv_kernel_v_read_readvariableop/savev2_adam_5th_conv_bias_v_read_readvariableop=savev2_adam_batch_normalization_4_gamma_v_read_readvariableop<savev2_adam_batch_normalization_4_beta_v_read_readvariableop1savev2_adam_6th_conv_kernel_v_read_readvariableop/savev2_adam_6th_conv_bias_v_read_readvariableop=savev2_adam_batch_normalization_5_gamma_v_read_readvariableop<savev2_adam_batch_normalization_5_beta_v_read_readvariableopAsavev2_adam_1st_fully_connected_of_0_kernel_v_read_readvariableop?savev2_adam_1st_fully_connected_of_0_bias_v_read_readvariableopAsavev2_adam_1st_fully_connected_of_1_kernel_v_read_readvariableop?savev2_adam_1st_fully_connected_of_1_bias_v_read_readvariableopAsavev2_adam_1st_fully_connected_of_2_kernel_v_read_readvariableop?savev2_adam_1st_fully_connected_of_2_bias_v_read_readvariableopAsavev2_adam_1st_fully_connected_of_3_kernel_v_read_readvariableop?savev2_adam_1st_fully_connected_of_3_bias_v_read_readvariableopAsavev2_adam_1st_fully_connected_of_4_kernel_v_read_readvariableop?savev2_adam_1st_fully_connected_of_4_bias_v_read_readvariableop1savev2_adam_output_0_kernel_v_read_readvariableop/savev2_adam_output_0_bias_v_read_readvariableop1savev2_adam_output_1_kernel_v_read_readvariableop/savev2_adam_output_1_bias_v_read_readvariableop1savev2_adam_output_2_kernel_v_read_readvariableop/savev2_adam_output_2_bias_v_read_readvariableop1savev2_adam_output_3_kernel_v_read_readvariableop/savev2_adam_output_3_bias_v_read_readvariableop1savev2_adam_output_4_kernel_v_read_readvariableop/savev2_adam_output_4_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *~
dtypest
r2p	?
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:?
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 f
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: Q

Identity_1IdentityIdentity:output:0^NoOp*
T0*
_output_shapes
: [
NoOpNoOp^MergeV2Checkpoints*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0*?
_input_shapes?
?: : : : : : : :  : : : : : :  : : : : : :	?$(:(:	?$(:(:	?$(:(:	?$(:(:	?$(:(:($:$:($:$:($:$:($:$:($:$: : : : : : : : : : : : : :  : : : :  : : : :	?$(:(:	?$(:(:	?$(:(:	?$(:(:	?$(:(:($:$:($:$:($:$:($:$:($:$: : : : :  : : : :  : : : :	?$(:(:	?$(:(:	?$(:(:	?$(:(:	?$(:(:($:$:($:$:($:$:($:$:($:$: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:,(
&
_output_shapes
: : 

_output_shapes
: : 

_output_shapes
: : 

_output_shapes
: : 

_output_shapes
: : 

_output_shapes
: :,(
&
_output_shapes
:  : 

_output_shapes
: : 	

_output_shapes
: : 


_output_shapes
: : 

_output_shapes
: : 

_output_shapes
: :,(
&
_output_shapes
:  : 

_output_shapes
: : 

_output_shapes
: : 

_output_shapes
: : 

_output_shapes
: : 

_output_shapes
: :%!

_output_shapes
:	?$(: 

_output_shapes
:(:%!

_output_shapes
:	?$(: 

_output_shapes
:(:%!

_output_shapes
:	?$(: 

_output_shapes
:(:%!

_output_shapes
:	?$(: 

_output_shapes
:(:%!

_output_shapes
:	?$(: 

_output_shapes
:(:$ 

_output_shapes

:($: 

_output_shapes
:$:$ 

_output_shapes

:($:  

_output_shapes
:$:$! 

_output_shapes

:($: "

_output_shapes
:$:$# 

_output_shapes

:($: $

_output_shapes
:$:$% 

_output_shapes

:($: &

_output_shapes
:$:'

_output_shapes
: :(

_output_shapes
: :)

_output_shapes
: :*

_output_shapes
: :+

_output_shapes
: :,

_output_shapes
: :-

_output_shapes
: :.

_output_shapes
: :/

_output_shapes
: :,0(
&
_output_shapes
: : 1

_output_shapes
: : 2

_output_shapes
: : 3

_output_shapes
: :,4(
&
_output_shapes
:  : 5

_output_shapes
: : 6

_output_shapes
: : 7

_output_shapes
: :,8(
&
_output_shapes
:  : 9

_output_shapes
: : :

_output_shapes
: : ;

_output_shapes
: :%<!

_output_shapes
:	?$(: =

_output_shapes
:(:%>!

_output_shapes
:	?$(: ?

_output_shapes
:(:%@!

_output_shapes
:	?$(: A

_output_shapes
:(:%B!

_output_shapes
:	?$(: C

_output_shapes
:(:%D!

_output_shapes
:	?$(: E

_output_shapes
:(:$F 

_output_shapes

:($: G

_output_shapes
:$:$H 

_output_shapes

:($: I

_output_shapes
:$:$J 

_output_shapes

:($: K

_output_shapes
:$:$L 

_output_shapes

:($: M

_output_shapes
:$:$N 

_output_shapes

:($: O

_output_shapes
:$:,P(
&
_output_shapes
: : Q

_output_shapes
: : R

_output_shapes
: : S

_output_shapes
: :,T(
&
_output_shapes
:  : U

_output_shapes
: : V

_output_shapes
: : W

_output_shapes
: :,X(
&
_output_shapes
:  : Y

_output_shapes
: : Z

_output_shapes
: : [

_output_shapes
: :%\!

_output_shapes
:	?$(: ]

_output_shapes
:(:%^!

_output_shapes
:	?$(: _

_output_shapes
:(:%`!

_output_shapes
:	?$(: a

_output_shapes
:(:%b!

_output_shapes
:	?$(: c

_output_shapes
:(:%d!

_output_shapes
:	?$(: e

_output_shapes
:(:$f 

_output_shapes

:($: g

_output_shapes
:$:$h 

_output_shapes

:($: i

_output_shapes
:$:$j 

_output_shapes

:($: k

_output_shapes
:$:$l 

_output_shapes

:($: m

_output_shapes
:$:$n 

_output_shapes

:($: o

_output_shapes
:$:p

_output_shapes
: 
?

?
D__inference_output_4_layer_call_and_return_conditional_losses_108474

inputs0
matmul_readvariableop_resource:($-
biasadd_readvariableop_resource:$
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:($*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????$r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:$*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????$V
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:?????????$`
IdentityIdentitySoftmax:softmax:0^NoOp*
T0*'
_output_shapes
:?????????$w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????(: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????(
 
_user_specified_nameinputs
?	
d
E__inference_dropout_2_layer_call_and_return_conditional_losses_108320

inputs
identity?R
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??d
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:?????????(C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:?????????(*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????(o
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????(i
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:?????????(Y
IdentityIdentitydropout/Mul_1:z:0*
T0*'
_output_shapes
:?????????("
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:?????????(:O K
'
_output_shapes
:?????????(
 
_user_specified_nameinputs
?	
?
6__inference_batch_normalization_3_layer_call_fn_107895

inputs
unknown: 
	unknown_0: 
	unknown_1: 
	unknown_2: 
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+??????????????????????????? *&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Z
fURS
Q__inference_batch_normalization_3_layer_call_and_return_conditional_losses_105840?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+??????????????????????????? `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+??????????????????????????? : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs
?
?
)__inference_4th_conv_layer_call_fn_107861

inputs!
unknown: 
	unknown_0: 
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????dd *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_4th_conv_layer_call_and_return_conditional_losses_106052w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????dd `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????dd: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????dd
 
_user_specified_nameinputs
?
?
)__inference_6th_conv_layer_call_fn_108045

inputs!
unknown:  
	unknown_0: 
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_6th_conv_layer_call_and_return_conditional_losses_106106w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:????????? `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:????????? : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:????????? 
 
_user_specified_nameinputs
??
?$
!__inference__wrapped_model_105806

imagesG
-model_4th_conv_conv2d_readvariableop_resource: <
.model_4th_conv_biasadd_readvariableop_resource: A
3model_batch_normalization_3_readvariableop_resource: C
5model_batch_normalization_3_readvariableop_1_resource: R
Dmodel_batch_normalization_3_fusedbatchnormv3_readvariableop_resource: T
Fmodel_batch_normalization_3_fusedbatchnormv3_readvariableop_1_resource: G
-model_5th_conv_conv2d_readvariableop_resource:  <
.model_5th_conv_biasadd_readvariableop_resource: A
3model_batch_normalization_4_readvariableop_resource: C
5model_batch_normalization_4_readvariableop_1_resource: R
Dmodel_batch_normalization_4_fusedbatchnormv3_readvariableop_resource: T
Fmodel_batch_normalization_4_fusedbatchnormv3_readvariableop_1_resource: G
-model_6th_conv_conv2d_readvariableop_resource:  <
.model_6th_conv_biasadd_readvariableop_resource: A
3model_batch_normalization_5_readvariableop_resource: C
5model_batch_normalization_5_readvariableop_1_resource: R
Dmodel_batch_normalization_5_fusedbatchnormv3_readvariableop_resource: T
Fmodel_batch_normalization_5_fusedbatchnormv3_readvariableop_1_resource: P
=model_1st_fully_connected_of_4_matmul_readvariableop_resource:	?$(L
>model_1st_fully_connected_of_4_biasadd_readvariableop_resource:(P
=model_1st_fully_connected_of_3_matmul_readvariableop_resource:	?$(L
>model_1st_fully_connected_of_3_biasadd_readvariableop_resource:(P
=model_1st_fully_connected_of_2_matmul_readvariableop_resource:	?$(L
>model_1st_fully_connected_of_2_biasadd_readvariableop_resource:(P
=model_1st_fully_connected_of_1_matmul_readvariableop_resource:	?$(L
>model_1st_fully_connected_of_1_biasadd_readvariableop_resource:(P
=model_1st_fully_connected_of_0_matmul_readvariableop_resource:	?$(L
>model_1st_fully_connected_of_0_biasadd_readvariableop_resource:(?
-model_output_0_matmul_readvariableop_resource:($<
.model_output_0_biasadd_readvariableop_resource:$?
-model_output_1_matmul_readvariableop_resource:($<
.model_output_1_biasadd_readvariableop_resource:$?
-model_output_2_matmul_readvariableop_resource:($<
.model_output_2_biasadd_readvariableop_resource:$?
-model_output_3_matmul_readvariableop_resource:($<
.model_output_3_biasadd_readvariableop_resource:$?
-model_output_4_matmul_readvariableop_resource:($<
.model_output_4_biasadd_readvariableop_resource:$
identity??5model/1st_fully_connected_of_0/BiasAdd/ReadVariableOp?4model/1st_fully_connected_of_0/MatMul/ReadVariableOp?5model/1st_fully_connected_of_1/BiasAdd/ReadVariableOp?4model/1st_fully_connected_of_1/MatMul/ReadVariableOp?5model/1st_fully_connected_of_2/BiasAdd/ReadVariableOp?4model/1st_fully_connected_of_2/MatMul/ReadVariableOp?5model/1st_fully_connected_of_3/BiasAdd/ReadVariableOp?4model/1st_fully_connected_of_3/MatMul/ReadVariableOp?5model/1st_fully_connected_of_4/BiasAdd/ReadVariableOp?4model/1st_fully_connected_of_4/MatMul/ReadVariableOp?%model/4th_conv/BiasAdd/ReadVariableOp?$model/4th_conv/Conv2D/ReadVariableOp?%model/5th_conv/BiasAdd/ReadVariableOp?$model/5th_conv/Conv2D/ReadVariableOp?%model/6th_conv/BiasAdd/ReadVariableOp?$model/6th_conv/Conv2D/ReadVariableOp?;model/batch_normalization_3/FusedBatchNormV3/ReadVariableOp?=model/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1?*model/batch_normalization_3/ReadVariableOp?,model/batch_normalization_3/ReadVariableOp_1?;model/batch_normalization_4/FusedBatchNormV3/ReadVariableOp?=model/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1?*model/batch_normalization_4/ReadVariableOp?,model/batch_normalization_4/ReadVariableOp_1?;model/batch_normalization_5/FusedBatchNormV3/ReadVariableOp?=model/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1?*model/batch_normalization_5/ReadVariableOp?,model/batch_normalization_5/ReadVariableOp_1?%model/output_0/BiasAdd/ReadVariableOp?$model/output_0/MatMul/ReadVariableOp?%model/output_1/BiasAdd/ReadVariableOp?$model/output_1/MatMul/ReadVariableOp?%model/output_2/BiasAdd/ReadVariableOp?$model/output_2/MatMul/ReadVariableOp?%model/output_3/BiasAdd/ReadVariableOp?$model/output_3/MatMul/ReadVariableOp?%model/output_4/BiasAdd/ReadVariableOp?$model/output_4/MatMul/ReadVariableOp?
$model/4th_conv/Conv2D/ReadVariableOpReadVariableOp-model_4th_conv_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0?
model/4th_conv/Conv2DConv2Dimages,model/4th_conv/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????dd *
paddingSAME*
strides
?
%model/4th_conv/BiasAdd/ReadVariableOpReadVariableOp.model_4th_conv_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0?
model/4th_conv/BiasAddBiasAddmodel/4th_conv/Conv2D:output:0-model/4th_conv/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????dd v
model/4th_conv/ReluRelumodel/4th_conv/BiasAdd:output:0*
T0*/
_output_shapes
:?????????dd ?
model/4th_pool/MaxPoolMaxPool!model/4th_conv/Relu:activations:0*/
_output_shapes
:?????????22 *
ksize
*
paddingVALID*
strides
?
*model/batch_normalization_3/ReadVariableOpReadVariableOp3model_batch_normalization_3_readvariableop_resource*
_output_shapes
: *
dtype0?
,model/batch_normalization_3/ReadVariableOp_1ReadVariableOp5model_batch_normalization_3_readvariableop_1_resource*
_output_shapes
: *
dtype0?
;model/batch_normalization_3/FusedBatchNormV3/ReadVariableOpReadVariableOpDmodel_batch_normalization_3_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0?
=model/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpFmodel_batch_normalization_3_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0?
,model/batch_normalization_3/FusedBatchNormV3FusedBatchNormV3model/4th_pool/MaxPool:output:02model/batch_normalization_3/ReadVariableOp:value:04model/batch_normalization_3/ReadVariableOp_1:value:0Cmodel/batch_normalization_3/FusedBatchNormV3/ReadVariableOp:value:0Emodel/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????22 : : : : :*
epsilon%o?:*
is_training( ?
$model/5th_conv/Conv2D/ReadVariableOpReadVariableOp-model_5th_conv_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype0?
model/5th_conv/Conv2DConv2D0model/batch_normalization_3/FusedBatchNormV3:y:0,model/5th_conv/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????22 *
paddingSAME*
strides
?
%model/5th_conv/BiasAdd/ReadVariableOpReadVariableOp.model_5th_conv_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0?
model/5th_conv/BiasAddBiasAddmodel/5th_conv/Conv2D:output:0-model/5th_conv/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????22 v
model/5th_conv/ReluRelumodel/5th_conv/BiasAdd:output:0*
T0*/
_output_shapes
:?????????22 ?
model/5th_pool/MaxPoolMaxPool!model/5th_conv/Relu:activations:0*/
_output_shapes
:????????? *
ksize
*
paddingVALID*
strides
?
*model/batch_normalization_4/ReadVariableOpReadVariableOp3model_batch_normalization_4_readvariableop_resource*
_output_shapes
: *
dtype0?
,model/batch_normalization_4/ReadVariableOp_1ReadVariableOp5model_batch_normalization_4_readvariableop_1_resource*
_output_shapes
: *
dtype0?
;model/batch_normalization_4/FusedBatchNormV3/ReadVariableOpReadVariableOpDmodel_batch_normalization_4_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0?
=model/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpFmodel_batch_normalization_4_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0?
,model/batch_normalization_4/FusedBatchNormV3FusedBatchNormV3model/5th_pool/MaxPool:output:02model/batch_normalization_4/ReadVariableOp:value:04model/batch_normalization_4/ReadVariableOp_1:value:0Cmodel/batch_normalization_4/FusedBatchNormV3/ReadVariableOp:value:0Emodel/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:????????? : : : : :*
epsilon%o?:*
is_training( ?
$model/6th_conv/Conv2D/ReadVariableOpReadVariableOp-model_6th_conv_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype0?
model/6th_conv/Conv2DConv2D0model/batch_normalization_4/FusedBatchNormV3:y:0,model/6th_conv/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? *
paddingSAME*
strides
?
%model/6th_conv/BiasAdd/ReadVariableOpReadVariableOp.model_6th_conv_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0?
model/6th_conv/BiasAddBiasAddmodel/6th_conv/Conv2D:output:0-model/6th_conv/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? v
model/6th_conv/ReluRelumodel/6th_conv/BiasAdd:output:0*
T0*/
_output_shapes
:????????? ?
model/6th_pool/MaxPoolMaxPool!model/6th_conv/Relu:activations:0*/
_output_shapes
:????????? *
ksize
*
paddingVALID*
strides
?
*model/batch_normalization_5/ReadVariableOpReadVariableOp3model_batch_normalization_5_readvariableop_resource*
_output_shapes
: *
dtype0?
,model/batch_normalization_5/ReadVariableOp_1ReadVariableOp5model_batch_normalization_5_readvariableop_1_resource*
_output_shapes
: *
dtype0?
;model/batch_normalization_5/FusedBatchNormV3/ReadVariableOpReadVariableOpDmodel_batch_normalization_5_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0?
=model/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpFmodel_batch_normalization_5_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0?
,model/batch_normalization_5/FusedBatchNormV3FusedBatchNormV3model/6th_pool/MaxPool:output:02model/batch_normalization_5/ReadVariableOp:value:04model/batch_normalization_5/ReadVariableOp_1:value:0Cmodel/batch_normalization_5/FusedBatchNormV3/ReadVariableOp:value:0Emodel/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:????????? : : : : :*
epsilon%o?:*
is_training( a
model/flat/ConstConst*
_output_shapes
:*
dtype0*
valueB"????   ?
model/flat/ReshapeReshape0model/batch_normalization_5/FusedBatchNormV3:y:0model/flat/Const:output:0*
T0*(
_output_shapes
:??????????$?
4model/1st_fully_connected_of_4/MatMul/ReadVariableOpReadVariableOp=model_1st_fully_connected_of_4_matmul_readvariableop_resource*
_output_shapes
:	?$(*
dtype0?
%model/1st_fully_connected_of_4/MatMulMatMulmodel/flat/Reshape:output:0<model/1st_fully_connected_of_4/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????(?
5model/1st_fully_connected_of_4/BiasAdd/ReadVariableOpReadVariableOp>model_1st_fully_connected_of_4_biasadd_readvariableop_resource*
_output_shapes
:(*
dtype0?
&model/1st_fully_connected_of_4/BiasAddBiasAdd/model/1st_fully_connected_of_4/MatMul:product:0=model/1st_fully_connected_of_4/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????(?
#model/1st_fully_connected_of_4/ReluRelu/model/1st_fully_connected_of_4/BiasAdd:output:0*
T0*'
_output_shapes
:?????????(?
4model/1st_fully_connected_of_3/MatMul/ReadVariableOpReadVariableOp=model_1st_fully_connected_of_3_matmul_readvariableop_resource*
_output_shapes
:	?$(*
dtype0?
%model/1st_fully_connected_of_3/MatMulMatMulmodel/flat/Reshape:output:0<model/1st_fully_connected_of_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????(?
5model/1st_fully_connected_of_3/BiasAdd/ReadVariableOpReadVariableOp>model_1st_fully_connected_of_3_biasadd_readvariableop_resource*
_output_shapes
:(*
dtype0?
&model/1st_fully_connected_of_3/BiasAddBiasAdd/model/1st_fully_connected_of_3/MatMul:product:0=model/1st_fully_connected_of_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????(?
#model/1st_fully_connected_of_3/ReluRelu/model/1st_fully_connected_of_3/BiasAdd:output:0*
T0*'
_output_shapes
:?????????(?
4model/1st_fully_connected_of_2/MatMul/ReadVariableOpReadVariableOp=model_1st_fully_connected_of_2_matmul_readvariableop_resource*
_output_shapes
:	?$(*
dtype0?
%model/1st_fully_connected_of_2/MatMulMatMulmodel/flat/Reshape:output:0<model/1st_fully_connected_of_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????(?
5model/1st_fully_connected_of_2/BiasAdd/ReadVariableOpReadVariableOp>model_1st_fully_connected_of_2_biasadd_readvariableop_resource*
_output_shapes
:(*
dtype0?
&model/1st_fully_connected_of_2/BiasAddBiasAdd/model/1st_fully_connected_of_2/MatMul:product:0=model/1st_fully_connected_of_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????(?
#model/1st_fully_connected_of_2/ReluRelu/model/1st_fully_connected_of_2/BiasAdd:output:0*
T0*'
_output_shapes
:?????????(?
4model/1st_fully_connected_of_1/MatMul/ReadVariableOpReadVariableOp=model_1st_fully_connected_of_1_matmul_readvariableop_resource*
_output_shapes
:	?$(*
dtype0?
%model/1st_fully_connected_of_1/MatMulMatMulmodel/flat/Reshape:output:0<model/1st_fully_connected_of_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????(?
5model/1st_fully_connected_of_1/BiasAdd/ReadVariableOpReadVariableOp>model_1st_fully_connected_of_1_biasadd_readvariableop_resource*
_output_shapes
:(*
dtype0?
&model/1st_fully_connected_of_1/BiasAddBiasAdd/model/1st_fully_connected_of_1/MatMul:product:0=model/1st_fully_connected_of_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????(?
#model/1st_fully_connected_of_1/ReluRelu/model/1st_fully_connected_of_1/BiasAdd:output:0*
T0*'
_output_shapes
:?????????(?
4model/1st_fully_connected_of_0/MatMul/ReadVariableOpReadVariableOp=model_1st_fully_connected_of_0_matmul_readvariableop_resource*
_output_shapes
:	?$(*
dtype0?
%model/1st_fully_connected_of_0/MatMulMatMulmodel/flat/Reshape:output:0<model/1st_fully_connected_of_0/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????(?
5model/1st_fully_connected_of_0/BiasAdd/ReadVariableOpReadVariableOp>model_1st_fully_connected_of_0_biasadd_readvariableop_resource*
_output_shapes
:(*
dtype0?
&model/1st_fully_connected_of_0/BiasAddBiasAdd/model/1st_fully_connected_of_0/MatMul:product:0=model/1st_fully_connected_of_0/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????(?
#model/1st_fully_connected_of_0/ReluRelu/model/1st_fully_connected_of_0/BiasAdd:output:0*
T0*'
_output_shapes
:?????????(?
model/dropout_4/IdentityIdentity1model/1st_fully_connected_of_4/Relu:activations:0*
T0*'
_output_shapes
:?????????(?
model/dropout_3/IdentityIdentity1model/1st_fully_connected_of_3/Relu:activations:0*
T0*'
_output_shapes
:?????????(?
model/dropout_2/IdentityIdentity1model/1st_fully_connected_of_2/Relu:activations:0*
T0*'
_output_shapes
:?????????(?
model/dropout_1/IdentityIdentity1model/1st_fully_connected_of_1/Relu:activations:0*
T0*'
_output_shapes
:?????????(?
model/dropout/IdentityIdentity1model/1st_fully_connected_of_0/Relu:activations:0*
T0*'
_output_shapes
:?????????(?
$model/output_0/MatMul/ReadVariableOpReadVariableOp-model_output_0_matmul_readvariableop_resource*
_output_shapes

:($*
dtype0?
model/output_0/MatMulMatMulmodel/dropout/Identity:output:0,model/output_0/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????$?
%model/output_0/BiasAdd/ReadVariableOpReadVariableOp.model_output_0_biasadd_readvariableop_resource*
_output_shapes
:$*
dtype0?
model/output_0/BiasAddBiasAddmodel/output_0/MatMul:product:0-model/output_0/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????$t
model/output_0/SoftmaxSoftmaxmodel/output_0/BiasAdd:output:0*
T0*'
_output_shapes
:?????????$?
$model/output_1/MatMul/ReadVariableOpReadVariableOp-model_output_1_matmul_readvariableop_resource*
_output_shapes

:($*
dtype0?
model/output_1/MatMulMatMul!model/dropout_1/Identity:output:0,model/output_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????$?
%model/output_1/BiasAdd/ReadVariableOpReadVariableOp.model_output_1_biasadd_readvariableop_resource*
_output_shapes
:$*
dtype0?
model/output_1/BiasAddBiasAddmodel/output_1/MatMul:product:0-model/output_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????$t
model/output_1/SoftmaxSoftmaxmodel/output_1/BiasAdd:output:0*
T0*'
_output_shapes
:?????????$?
$model/output_2/MatMul/ReadVariableOpReadVariableOp-model_output_2_matmul_readvariableop_resource*
_output_shapes

:($*
dtype0?
model/output_2/MatMulMatMul!model/dropout_2/Identity:output:0,model/output_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????$?
%model/output_2/BiasAdd/ReadVariableOpReadVariableOp.model_output_2_biasadd_readvariableop_resource*
_output_shapes
:$*
dtype0?
model/output_2/BiasAddBiasAddmodel/output_2/MatMul:product:0-model/output_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????$t
model/output_2/SoftmaxSoftmaxmodel/output_2/BiasAdd:output:0*
T0*'
_output_shapes
:?????????$?
$model/output_3/MatMul/ReadVariableOpReadVariableOp-model_output_3_matmul_readvariableop_resource*
_output_shapes

:($*
dtype0?
model/output_3/MatMulMatMul!model/dropout_3/Identity:output:0,model/output_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????$?
%model/output_3/BiasAdd/ReadVariableOpReadVariableOp.model_output_3_biasadd_readvariableop_resource*
_output_shapes
:$*
dtype0?
model/output_3/BiasAddBiasAddmodel/output_3/MatMul:product:0-model/output_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????$t
model/output_3/SoftmaxSoftmaxmodel/output_3/BiasAdd:output:0*
T0*'
_output_shapes
:?????????$?
$model/output_4/MatMul/ReadVariableOpReadVariableOp-model_output_4_matmul_readvariableop_resource*
_output_shapes

:($*
dtype0?
model/output_4/MatMulMatMul!model/dropout_4/Identity:output:0,model/output_4/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????$?
%model/output_4/BiasAdd/ReadVariableOpReadVariableOp.model_output_4_biasadd_readvariableop_resource*
_output_shapes
:$*
dtype0?
model/output_4/BiasAddBiasAddmodel/output_4/MatMul:product:0-model/output_4/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????$t
model/output_4/SoftmaxSoftmaxmodel/output_4/BiasAdd:output:0*
T0*'
_output_shapes
:?????????$?
!model/tf.convert_to_tensor_1/CastCast model/output_0/Softmax:softmax:0*

DstT0*

SrcT0*'
_output_shapes
:?????????$?
#model/tf.convert_to_tensor_1/Cast_1Cast model/output_1/Softmax:softmax:0*

DstT0*

SrcT0*'
_output_shapes
:?????????$?
#model/tf.convert_to_tensor_1/Cast_2Cast model/output_2/Softmax:softmax:0*

DstT0*

SrcT0*'
_output_shapes
:?????????$?
#model/tf.convert_to_tensor_1/Cast_3Cast model/output_3/Softmax:softmax:0*

DstT0*

SrcT0*'
_output_shapes
:?????????$?
#model/tf.convert_to_tensor_1/Cast_4Cast model/output_4/Softmax:softmax:0*

DstT0*

SrcT0*'
_output_shapes
:?????????$?
#model/tf.convert_to_tensor_1/packedPack%model/tf.convert_to_tensor_1/Cast:y:0'model/tf.convert_to_tensor_1/Cast_1:y:0'model/tf.convert_to_tensor_1/Cast_2:y:0'model/tf.convert_to_tensor_1/Cast_3:y:0'model/tf.convert_to_tensor_1/Cast_4:y:0*
N*
T0*+
_output_shapes
:?????????$?
-model/tf.compat.v1.transpose_1/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          ?
(model/tf.compat.v1.transpose_1/transpose	Transpose,model/tf.convert_to_tensor_1/packed:output:06model/tf.compat.v1.transpose_1/transpose/perm:output:0*
T0*+
_output_shapes
:?????????$
IdentityIdentity,model/tf.compat.v1.transpose_1/transpose:y:0^NoOp*
T0*+
_output_shapes
:?????????$?
NoOpNoOp6^model/1st_fully_connected_of_0/BiasAdd/ReadVariableOp5^model/1st_fully_connected_of_0/MatMul/ReadVariableOp6^model/1st_fully_connected_of_1/BiasAdd/ReadVariableOp5^model/1st_fully_connected_of_1/MatMul/ReadVariableOp6^model/1st_fully_connected_of_2/BiasAdd/ReadVariableOp5^model/1st_fully_connected_of_2/MatMul/ReadVariableOp6^model/1st_fully_connected_of_3/BiasAdd/ReadVariableOp5^model/1st_fully_connected_of_3/MatMul/ReadVariableOp6^model/1st_fully_connected_of_4/BiasAdd/ReadVariableOp5^model/1st_fully_connected_of_4/MatMul/ReadVariableOp&^model/4th_conv/BiasAdd/ReadVariableOp%^model/4th_conv/Conv2D/ReadVariableOp&^model/5th_conv/BiasAdd/ReadVariableOp%^model/5th_conv/Conv2D/ReadVariableOp&^model/6th_conv/BiasAdd/ReadVariableOp%^model/6th_conv/Conv2D/ReadVariableOp<^model/batch_normalization_3/FusedBatchNormV3/ReadVariableOp>^model/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1+^model/batch_normalization_3/ReadVariableOp-^model/batch_normalization_3/ReadVariableOp_1<^model/batch_normalization_4/FusedBatchNormV3/ReadVariableOp>^model/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1+^model/batch_normalization_4/ReadVariableOp-^model/batch_normalization_4/ReadVariableOp_1<^model/batch_normalization_5/FusedBatchNormV3/ReadVariableOp>^model/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1+^model/batch_normalization_5/ReadVariableOp-^model/batch_normalization_5/ReadVariableOp_1&^model/output_0/BiasAdd/ReadVariableOp%^model/output_0/MatMul/ReadVariableOp&^model/output_1/BiasAdd/ReadVariableOp%^model/output_1/MatMul/ReadVariableOp&^model/output_2/BiasAdd/ReadVariableOp%^model/output_2/MatMul/ReadVariableOp&^model/output_3/BiasAdd/ReadVariableOp%^model/output_3/MatMul/ReadVariableOp&^model/output_4/BiasAdd/ReadVariableOp%^model/output_4/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*z
_input_shapesi
g:?????????dd: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2n
5model/1st_fully_connected_of_0/BiasAdd/ReadVariableOp5model/1st_fully_connected_of_0/BiasAdd/ReadVariableOp2l
4model/1st_fully_connected_of_0/MatMul/ReadVariableOp4model/1st_fully_connected_of_0/MatMul/ReadVariableOp2n
5model/1st_fully_connected_of_1/BiasAdd/ReadVariableOp5model/1st_fully_connected_of_1/BiasAdd/ReadVariableOp2l
4model/1st_fully_connected_of_1/MatMul/ReadVariableOp4model/1st_fully_connected_of_1/MatMul/ReadVariableOp2n
5model/1st_fully_connected_of_2/BiasAdd/ReadVariableOp5model/1st_fully_connected_of_2/BiasAdd/ReadVariableOp2l
4model/1st_fully_connected_of_2/MatMul/ReadVariableOp4model/1st_fully_connected_of_2/MatMul/ReadVariableOp2n
5model/1st_fully_connected_of_3/BiasAdd/ReadVariableOp5model/1st_fully_connected_of_3/BiasAdd/ReadVariableOp2l
4model/1st_fully_connected_of_3/MatMul/ReadVariableOp4model/1st_fully_connected_of_3/MatMul/ReadVariableOp2n
5model/1st_fully_connected_of_4/BiasAdd/ReadVariableOp5model/1st_fully_connected_of_4/BiasAdd/ReadVariableOp2l
4model/1st_fully_connected_of_4/MatMul/ReadVariableOp4model/1st_fully_connected_of_4/MatMul/ReadVariableOp2N
%model/4th_conv/BiasAdd/ReadVariableOp%model/4th_conv/BiasAdd/ReadVariableOp2L
$model/4th_conv/Conv2D/ReadVariableOp$model/4th_conv/Conv2D/ReadVariableOp2N
%model/5th_conv/BiasAdd/ReadVariableOp%model/5th_conv/BiasAdd/ReadVariableOp2L
$model/5th_conv/Conv2D/ReadVariableOp$model/5th_conv/Conv2D/ReadVariableOp2N
%model/6th_conv/BiasAdd/ReadVariableOp%model/6th_conv/BiasAdd/ReadVariableOp2L
$model/6th_conv/Conv2D/ReadVariableOp$model/6th_conv/Conv2D/ReadVariableOp2z
;model/batch_normalization_3/FusedBatchNormV3/ReadVariableOp;model/batch_normalization_3/FusedBatchNormV3/ReadVariableOp2~
=model/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1=model/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_12X
*model/batch_normalization_3/ReadVariableOp*model/batch_normalization_3/ReadVariableOp2\
,model/batch_normalization_3/ReadVariableOp_1,model/batch_normalization_3/ReadVariableOp_12z
;model/batch_normalization_4/FusedBatchNormV3/ReadVariableOp;model/batch_normalization_4/FusedBatchNormV3/ReadVariableOp2~
=model/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1=model/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_12X
*model/batch_normalization_4/ReadVariableOp*model/batch_normalization_4/ReadVariableOp2\
,model/batch_normalization_4/ReadVariableOp_1,model/batch_normalization_4/ReadVariableOp_12z
;model/batch_normalization_5/FusedBatchNormV3/ReadVariableOp;model/batch_normalization_5/FusedBatchNormV3/ReadVariableOp2~
=model/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1=model/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_12X
*model/batch_normalization_5/ReadVariableOp*model/batch_normalization_5/ReadVariableOp2\
,model/batch_normalization_5/ReadVariableOp_1,model/batch_normalization_5/ReadVariableOp_12N
%model/output_0/BiasAdd/ReadVariableOp%model/output_0/BiasAdd/ReadVariableOp2L
$model/output_0/MatMul/ReadVariableOp$model/output_0/MatMul/ReadVariableOp2N
%model/output_1/BiasAdd/ReadVariableOp%model/output_1/BiasAdd/ReadVariableOp2L
$model/output_1/MatMul/ReadVariableOp$model/output_1/MatMul/ReadVariableOp2N
%model/output_2/BiasAdd/ReadVariableOp%model/output_2/BiasAdd/ReadVariableOp2L
$model/output_2/MatMul/ReadVariableOp$model/output_2/MatMul/ReadVariableOp2N
%model/output_3/BiasAdd/ReadVariableOp%model/output_3/BiasAdd/ReadVariableOp2L
$model/output_3/MatMul/ReadVariableOp$model/output_3/MatMul/ReadVariableOp2N
%model/output_4/BiasAdd/ReadVariableOp%model/output_4/BiasAdd/ReadVariableOp2L
$model/output_4/MatMul/ReadVariableOp$model/output_4/MatMul/ReadVariableOp:W S
/
_output_shapes
:?????????dd
 
_user_specified_nameimages
?
c
*__inference_dropout_4_layer_call_fn_108357

inputs
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????(* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dropout_4_layer_call_and_return_conditional_losses_106585o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????(`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:?????????(22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????(
 
_user_specified_nameinputs
?	
d
E__inference_dropout_1_layer_call_and_return_conditional_losses_106516

inputs
identity?R
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??d
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:?????????(C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:?????????(*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????(o
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????(i
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:?????????(Y
IdentityIdentitydropout/Mul_1:z:0*
T0*'
_output_shapes
:?????????("
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:?????????(:O K
'
_output_shapes
:?????????(
 
_user_specified_nameinputs
?	
?
6__inference_batch_normalization_3_layer_call_fn_107908

inputs
unknown: 
	unknown_0: 
	unknown_1: 
	unknown_2: 
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+??????????????????????????? *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Z
fURS
Q__inference_batch_normalization_3_layer_call_and_return_conditional_losses_105871?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+??????????????????????????? `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+??????????????????????????? : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs
?
?
Q__inference_batch_normalization_4_layer_call_and_return_conditional_losses_105916

inputs%
readvariableop_resource: '
readvariableop_1_resource: 6
(fusedbatchnormv3_readvariableop_resource: 8
*fusedbatchnormv3_readvariableop_1_resource: 
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype0?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+??????????????????????????? : : : : :*
epsilon%o?:*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+??????????????????????????? ?
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+??????????????????????????? : : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs
?
c
*__inference_dropout_1_layer_call_fn_108276

inputs
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????(* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dropout_1_layer_call_and_return_conditional_losses_106516o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????(`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:?????????(22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????(
 
_user_specified_nameinputs
?

?
D__inference_output_1_layer_call_and_return_conditional_losses_106278

inputs0
matmul_readvariableop_resource:($-
biasadd_readvariableop_resource:$
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:($*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????$r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:$*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????$V
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:?????????$`
IdentityIdentitySoftmax:softmax:0^NoOp*
T0*'
_output_shapes
:?????????$w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????(: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????(
 
_user_specified_nameinputs
?	
d
E__inference_dropout_3_layer_call_and_return_conditional_losses_106562

inputs
identity?R
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??d
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:?????????(C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:?????????(*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????(o
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????(i
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:?????????(Y
IdentityIdentitydropout/Mul_1:z:0*
T0*'
_output_shapes
:?????????("
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:?????????(:O K
'
_output_shapes
:?????????(
 
_user_specified_nameinputs
Ԉ
?
A__inference_model_layer_call_and_return_conditional_losses_107256

images(
th_conv_107146: 
th_conv_107148: *
batch_normalization_3_107152: *
batch_normalization_3_107154: *
batch_normalization_3_107156: *
batch_normalization_3_107158: (
th_conv_107161:  
th_conv_107163: *
batch_normalization_4_107167: *
batch_normalization_4_107169: *
batch_normalization_4_107171: *
batch_normalization_4_107173: (
th_conv_107176:  
th_conv_107178: *
batch_normalization_5_107182: *
batch_normalization_5_107184: *
batch_normalization_5_107186: *
batch_normalization_5_107188: 1
st_fully_connected_of_4_107192:	?$(,
st_fully_connected_of_4_107194:(1
st_fully_connected_of_3_107197:	?$(,
st_fully_connected_of_3_107199:(1
st_fully_connected_of_2_107202:	?$(,
st_fully_connected_of_2_107204:(1
st_fully_connected_of_1_107207:	?$(,
st_fully_connected_of_1_107209:(1
st_fully_connected_of_0_107212:	?$(,
st_fully_connected_of_0_107214:(!
output_0_107222:($
output_0_107224:$!
output_1_107227:($
output_1_107229:$!
output_2_107232:($
output_2_107234:$!
output_3_107237:($
output_3_107239:$!
output_4_107242:($
output_4_107244:$
identity??01st_fully_connected_of_0/StatefulPartitionedCall?01st_fully_connected_of_1/StatefulPartitionedCall?01st_fully_connected_of_2/StatefulPartitionedCall?01st_fully_connected_of_3/StatefulPartitionedCall?01st_fully_connected_of_4/StatefulPartitionedCall? 4th_conv/StatefulPartitionedCall? 5th_conv/StatefulPartitionedCall? 6th_conv/StatefulPartitionedCall?-batch_normalization_3/StatefulPartitionedCall?-batch_normalization_4/StatefulPartitionedCall?-batch_normalization_5/StatefulPartitionedCall?dropout/StatefulPartitionedCall?!dropout_1/StatefulPartitionedCall?!dropout_2/StatefulPartitionedCall?!dropout_3/StatefulPartitionedCall?!dropout_4/StatefulPartitionedCall? output_0/StatefulPartitionedCall? output_1/StatefulPartitionedCall? output_2/StatefulPartitionedCall? output_3/StatefulPartitionedCall? output_4/StatefulPartitionedCall?
 4th_conv/StatefulPartitionedCallStatefulPartitionedCallimagesth_conv_107146th_conv_107148*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????dd *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_4th_conv_layer_call_and_return_conditional_losses_106052?
4th_pool/PartitionedCallPartitionedCall)4th_conv/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????22 * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_4th_pool_layer_call_and_return_conditional_losses_105815?
-batch_normalization_3/StatefulPartitionedCallStatefulPartitionedCall!4th_pool/PartitionedCall:output:0batch_normalization_3_107152batch_normalization_3_107154batch_normalization_3_107156batch_normalization_3_107158*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????22 *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Z
fURS
Q__inference_batch_normalization_3_layer_call_and_return_conditional_losses_105871?
 5th_conv/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_3/StatefulPartitionedCall:output:0th_conv_107161th_conv_107163*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????22 *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_5th_conv_layer_call_and_return_conditional_losses_106079?
5th_pool/PartitionedCallPartitionedCall)5th_conv/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_5th_pool_layer_call_and_return_conditional_losses_105891?
-batch_normalization_4/StatefulPartitionedCallStatefulPartitionedCall!5th_pool/PartitionedCall:output:0batch_normalization_4_107167batch_normalization_4_107169batch_normalization_4_107171batch_normalization_4_107173*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Z
fURS
Q__inference_batch_normalization_4_layer_call_and_return_conditional_losses_105947?
 6th_conv/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_4/StatefulPartitionedCall:output:0th_conv_107176th_conv_107178*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_6th_conv_layer_call_and_return_conditional_losses_106106?
6th_pool/PartitionedCallPartitionedCall)6th_conv/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_6th_pool_layer_call_and_return_conditional_losses_105967?
-batch_normalization_5/StatefulPartitionedCallStatefulPartitionedCall!6th_pool/PartitionedCall:output:0batch_normalization_5_107182batch_normalization_5_107184batch_normalization_5_107186batch_normalization_5_107188*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Z
fURS
Q__inference_batch_normalization_5_layer_call_and_return_conditional_losses_106023?
flat/PartitionedCallPartitionedCall6batch_normalization_5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????$* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *I
fDRB
@__inference_flat_layer_call_and_return_conditional_losses_106128?
01st_fully_connected_of_4/StatefulPartitionedCallStatefulPartitionedCallflat/PartitionedCall:output:0st_fully_connected_of_4_107192st_fully_connected_of_4_107194*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????(*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *]
fXRV
T__inference_1st_fully_connected_of_4_layer_call_and_return_conditional_losses_106141?
01st_fully_connected_of_3/StatefulPartitionedCallStatefulPartitionedCallflat/PartitionedCall:output:0st_fully_connected_of_3_107197st_fully_connected_of_3_107199*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????(*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *]
fXRV
T__inference_1st_fully_connected_of_3_layer_call_and_return_conditional_losses_106158?
01st_fully_connected_of_2/StatefulPartitionedCallStatefulPartitionedCallflat/PartitionedCall:output:0st_fully_connected_of_2_107202st_fully_connected_of_2_107204*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????(*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *]
fXRV
T__inference_1st_fully_connected_of_2_layer_call_and_return_conditional_losses_106175?
01st_fully_connected_of_1/StatefulPartitionedCallStatefulPartitionedCallflat/PartitionedCall:output:0st_fully_connected_of_1_107207st_fully_connected_of_1_107209*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????(*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *]
fXRV
T__inference_1st_fully_connected_of_1_layer_call_and_return_conditional_losses_106192?
01st_fully_connected_of_0/StatefulPartitionedCallStatefulPartitionedCallflat/PartitionedCall:output:0st_fully_connected_of_0_107212st_fully_connected_of_0_107214*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????(*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *]
fXRV
T__inference_1st_fully_connected_of_0_layer_call_and_return_conditional_losses_106209?
!dropout_4/StatefulPartitionedCallStatefulPartitionedCall91st_fully_connected_of_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????(* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dropout_4_layer_call_and_return_conditional_losses_106585?
!dropout_3/StatefulPartitionedCallStatefulPartitionedCall91st_fully_connected_of_3/StatefulPartitionedCall:output:0"^dropout_4/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????(* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dropout_3_layer_call_and_return_conditional_losses_106562?
!dropout_2/StatefulPartitionedCallStatefulPartitionedCall91st_fully_connected_of_2/StatefulPartitionedCall:output:0"^dropout_3/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????(* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dropout_2_layer_call_and_return_conditional_losses_106539?
!dropout_1/StatefulPartitionedCallStatefulPartitionedCall91st_fully_connected_of_1/StatefulPartitionedCall:output:0"^dropout_2/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????(* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dropout_1_layer_call_and_return_conditional_losses_106516?
dropout/StatefulPartitionedCallStatefulPartitionedCall91st_fully_connected_of_0/StatefulPartitionedCall:output:0"^dropout_1/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????(* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dropout_layer_call_and_return_conditional_losses_106493?
 output_0/StatefulPartitionedCallStatefulPartitionedCall(dropout/StatefulPartitionedCall:output:0output_0_107222output_0_107224*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????$*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_output_0_layer_call_and_return_conditional_losses_106261?
 output_1/StatefulPartitionedCallStatefulPartitionedCall*dropout_1/StatefulPartitionedCall:output:0output_1_107227output_1_107229*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????$*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_output_1_layer_call_and_return_conditional_losses_106278?
 output_2/StatefulPartitionedCallStatefulPartitionedCall*dropout_2/StatefulPartitionedCall:output:0output_2_107232output_2_107234*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????$*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_output_2_layer_call_and_return_conditional_losses_106295?
 output_3/StatefulPartitionedCallStatefulPartitionedCall*dropout_3/StatefulPartitionedCall:output:0output_3_107237output_3_107239*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????$*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_output_3_layer_call_and_return_conditional_losses_106312?
 output_4/StatefulPartitionedCallStatefulPartitionedCall*dropout_4/StatefulPartitionedCall:output:0output_4_107242output_4_107244*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????$*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_output_4_layer_call_and_return_conditional_losses_106329?
tf.convert_to_tensor_1/CastCast)output_0/StatefulPartitionedCall:output:0*

DstT0*

SrcT0*'
_output_shapes
:?????????$?
tf.convert_to_tensor_1/Cast_1Cast)output_1/StatefulPartitionedCall:output:0*

DstT0*

SrcT0*'
_output_shapes
:?????????$?
tf.convert_to_tensor_1/Cast_2Cast)output_2/StatefulPartitionedCall:output:0*

DstT0*

SrcT0*'
_output_shapes
:?????????$?
tf.convert_to_tensor_1/Cast_3Cast)output_3/StatefulPartitionedCall:output:0*

DstT0*

SrcT0*'
_output_shapes
:?????????$?
tf.convert_to_tensor_1/Cast_4Cast)output_4/StatefulPartitionedCall:output:0*

DstT0*

SrcT0*'
_output_shapes
:?????????$?
tf.convert_to_tensor_1/packedPacktf.convert_to_tensor_1/Cast:y:0!tf.convert_to_tensor_1/Cast_1:y:0!tf.convert_to_tensor_1/Cast_2:y:0!tf.convert_to_tensor_1/Cast_3:y:0!tf.convert_to_tensor_1/Cast_4:y:0*
N*
T0*+
_output_shapes
:?????????$|
'tf.compat.v1.transpose_1/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          ?
"tf.compat.v1.transpose_1/transpose	Transpose&tf.convert_to_tensor_1/packed:output:00tf.compat.v1.transpose_1/transpose/perm:output:0*
T0*+
_output_shapes
:?????????$y
IdentityIdentity&tf.compat.v1.transpose_1/transpose:y:0^NoOp*
T0*+
_output_shapes
:?????????$?
NoOpNoOp1^1st_fully_connected_of_0/StatefulPartitionedCall1^1st_fully_connected_of_1/StatefulPartitionedCall1^1st_fully_connected_of_2/StatefulPartitionedCall1^1st_fully_connected_of_3/StatefulPartitionedCall1^1st_fully_connected_of_4/StatefulPartitionedCall!^4th_conv/StatefulPartitionedCall!^5th_conv/StatefulPartitionedCall!^6th_conv/StatefulPartitionedCall.^batch_normalization_3/StatefulPartitionedCall.^batch_normalization_4/StatefulPartitionedCall.^batch_normalization_5/StatefulPartitionedCall ^dropout/StatefulPartitionedCall"^dropout_1/StatefulPartitionedCall"^dropout_2/StatefulPartitionedCall"^dropout_3/StatefulPartitionedCall"^dropout_4/StatefulPartitionedCall!^output_0/StatefulPartitionedCall!^output_1/StatefulPartitionedCall!^output_2/StatefulPartitionedCall!^output_3/StatefulPartitionedCall!^output_4/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*z
_input_shapesi
g:?????????dd: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2d
01st_fully_connected_of_0/StatefulPartitionedCall01st_fully_connected_of_0/StatefulPartitionedCall2d
01st_fully_connected_of_1/StatefulPartitionedCall01st_fully_connected_of_1/StatefulPartitionedCall2d
01st_fully_connected_of_2/StatefulPartitionedCall01st_fully_connected_of_2/StatefulPartitionedCall2d
01st_fully_connected_of_3/StatefulPartitionedCall01st_fully_connected_of_3/StatefulPartitionedCall2d
01st_fully_connected_of_4/StatefulPartitionedCall01st_fully_connected_of_4/StatefulPartitionedCall2D
 4th_conv/StatefulPartitionedCall 4th_conv/StatefulPartitionedCall2D
 5th_conv/StatefulPartitionedCall 5th_conv/StatefulPartitionedCall2D
 6th_conv/StatefulPartitionedCall 6th_conv/StatefulPartitionedCall2^
-batch_normalization_3/StatefulPartitionedCall-batch_normalization_3/StatefulPartitionedCall2^
-batch_normalization_4/StatefulPartitionedCall-batch_normalization_4/StatefulPartitionedCall2^
-batch_normalization_5/StatefulPartitionedCall-batch_normalization_5/StatefulPartitionedCall2B
dropout/StatefulPartitionedCalldropout/StatefulPartitionedCall2F
!dropout_1/StatefulPartitionedCall!dropout_1/StatefulPartitionedCall2F
!dropout_2/StatefulPartitionedCall!dropout_2/StatefulPartitionedCall2F
!dropout_3/StatefulPartitionedCall!dropout_3/StatefulPartitionedCall2F
!dropout_4/StatefulPartitionedCall!dropout_4/StatefulPartitionedCall2D
 output_0/StatefulPartitionedCall output_0/StatefulPartitionedCall2D
 output_1/StatefulPartitionedCall output_1/StatefulPartitionedCall2D
 output_2/StatefulPartitionedCall output_2/StatefulPartitionedCall2D
 output_3/StatefulPartitionedCall output_3/StatefulPartitionedCall2D
 output_4/StatefulPartitionedCall output_4/StatefulPartitionedCall:W S
/
_output_shapes
:?????????dd
 
_user_specified_nameimages
?

?
T__inference_1st_fully_connected_of_1_layer_call_and_return_conditional_losses_108179

inputs1
matmul_readvariableop_resource:	?$(-
biasadd_readvariableop_resource:(
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?$(*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????(r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:(*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????(P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:?????????(a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:?????????(w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????$: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????$
 
_user_specified_nameinputs
?
E
)__inference_4th_pool_layer_call_fn_107877

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4????????????????????????????????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_4th_pool_layer_call_and_return_conditional_losses_105815?
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?

?
D__inference_output_3_layer_call_and_return_conditional_losses_106312

inputs0
matmul_readvariableop_resource:($-
biasadd_readvariableop_resource:$
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:($*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????$r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:$*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????$V
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:?????????$`
IdentityIdentitySoftmax:softmax:0^NoOp*
T0*'
_output_shapes
:?????????$w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????(: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????(
 
_user_specified_nameinputs
?
c
E__inference_dropout_1_layer_call_and_return_conditional_losses_108281

inputs

identity_1N
IdentityIdentityinputs*
T0*'
_output_shapes
:?????????([

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:?????????("!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:?????????(:O K
'
_output_shapes
:?????????(
 
_user_specified_nameinputs
?
a
(__inference_dropout_layer_call_fn_108249

inputs
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????(* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dropout_layer_call_and_return_conditional_losses_106493o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????(`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:?????????(22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????(
 
_user_specified_nameinputs
?
`
D__inference_5th_pool_layer_call_and_return_conditional_losses_105891

inputs
identity?
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4????????????????????????????????????*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
a
C__inference_dropout_layer_call_and_return_conditional_losses_106248

inputs

identity_1N
IdentityIdentityinputs*
T0*'
_output_shapes
:?????????([

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:?????????("!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:?????????(:O K
'
_output_shapes
:?????????(
 
_user_specified_nameinputs
?
?
D__inference_6th_conv_layer_call_and_return_conditional_losses_106106

inputs8
conv2d_readvariableop_resource:  -
biasadd_readvariableop_resource: 
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:  *
dtype0?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? *
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:????????? i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:????????? w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:????????? : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:????????? 
 
_user_specified_nameinputs
?	
d
E__inference_dropout_3_layer_call_and_return_conditional_losses_108347

inputs
identity?R
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??d
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:?????????(C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:?????????(*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????(o
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????(i
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:?????????(Y
IdentityIdentitydropout/Mul_1:z:0*
T0*'
_output_shapes
:?????????("
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:?????????(:O K
'
_output_shapes
:?????????(
 
_user_specified_nameinputs
?
?
D__inference_5th_conv_layer_call_and_return_conditional_losses_106079

inputs8
conv2d_readvariableop_resource:  -
biasadd_readvariableop_resource: 
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:  *
dtype0?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????22 *
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????22 X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:?????????22 i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:?????????22 w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????22 : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????22 
 
_user_specified_nameinputs
?	
d
E__inference_dropout_4_layer_call_and_return_conditional_losses_108374

inputs
identity?R
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??d
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:?????????(C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:?????????(*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????(o
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????(i
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:?????????(Y
IdentityIdentitydropout/Mul_1:z:0*
T0*'
_output_shapes
:?????????("
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:?????????(:O K
'
_output_shapes
:?????????(
 
_user_specified_nameinputs
?
?
Q__inference_batch_normalization_5_layer_call_and_return_conditional_losses_105992

inputs%
readvariableop_resource: '
readvariableop_1_resource: 6
(fusedbatchnormv3_readvariableop_resource: 8
*fusedbatchnormv3_readvariableop_1_resource: 
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype0?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+??????????????????????????? : : : : :*
epsilon%o?:*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+??????????????????????????? ?
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+??????????????????????????? : : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs
?

?
D__inference_output_3_layer_call_and_return_conditional_losses_108454

inputs0
matmul_readvariableop_resource:($-
biasadd_readvariableop_resource:$
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:($*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????$r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:$*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????$V
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:?????????$`
IdentityIdentitySoftmax:softmax:0^NoOp*
T0*'
_output_shapes
:?????????$w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????(: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????(
 
_user_specified_nameinputs
?
?
)__inference_output_0_layer_call_fn_108383

inputs
unknown:($
	unknown_0:$
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????$*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_output_0_layer_call_and_return_conditional_losses_106261o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????$`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????(: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????(
 
_user_specified_nameinputs
?
?
Q__inference_batch_normalization_5_layer_call_and_return_conditional_losses_108128

inputs%
readvariableop_resource: '
readvariableop_1_resource: 6
(fusedbatchnormv3_readvariableop_resource: 8
*fusedbatchnormv3_readvariableop_1_resource: 
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype0?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+??????????????????????????? : : : : :*
epsilon%o?:*
exponential_avg_factor%
?#<?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(}
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+??????????????????????????? ?
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+??????????????????????????? : : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs
?
?
$__inference_signature_wrapper_107345

images!
unknown: 
	unknown_0: 
	unknown_1: 
	unknown_2: 
	unknown_3: 
	unknown_4: #
	unknown_5:  
	unknown_6: 
	unknown_7: 
	unknown_8: 
	unknown_9: 

unknown_10: $

unknown_11:  

unknown_12: 

unknown_13: 

unknown_14: 

unknown_15: 

unknown_16: 

unknown_17:	?$(

unknown_18:(

unknown_19:	?$(

unknown_20:(

unknown_21:	?$(

unknown_22:(

unknown_23:	?$(

unknown_24:(

unknown_25:	?$(

unknown_26:(

unknown_27:($

unknown_28:$

unknown_29:($

unknown_30:$

unknown_31:($

unknown_32:$

unknown_33:($

unknown_34:$

unknown_35:($

unknown_36:$
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallimagesunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36*2
Tin+
)2'*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????$*H
_read_only_resource_inputs*
(&	
 !"#$%&*-
config_proto

CPU

GPU 2J 8? **
f%R#
!__inference__wrapped_model_105806s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:?????????$`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*z
_input_shapesi
g:?????????dd: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????dd
 
_user_specified_nameimages
?

?
T__inference_1st_fully_connected_of_1_layer_call_and_return_conditional_losses_106192

inputs1
matmul_readvariableop_resource:	?$(-
biasadd_readvariableop_resource:(
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?$(*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????(r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:(*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????(P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:?????????(a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:?????????(w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????$: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????$
 
_user_specified_nameinputs
??
?
A__inference_model_layer_call_and_return_conditional_losses_106344

inputs(
th_conv_106053: 
th_conv_106055: *
batch_normalization_3_106059: *
batch_normalization_3_106061: *
batch_normalization_3_106063: *
batch_normalization_3_106065: (
th_conv_106080:  
th_conv_106082: *
batch_normalization_4_106086: *
batch_normalization_4_106088: *
batch_normalization_4_106090: *
batch_normalization_4_106092: (
th_conv_106107:  
th_conv_106109: *
batch_normalization_5_106113: *
batch_normalization_5_106115: *
batch_normalization_5_106117: *
batch_normalization_5_106119: 1
st_fully_connected_of_4_106142:	?$(,
st_fully_connected_of_4_106144:(1
st_fully_connected_of_3_106159:	?$(,
st_fully_connected_of_3_106161:(1
st_fully_connected_of_2_106176:	?$(,
st_fully_connected_of_2_106178:(1
st_fully_connected_of_1_106193:	?$(,
st_fully_connected_of_1_106195:(1
st_fully_connected_of_0_106210:	?$(,
st_fully_connected_of_0_106212:(!
output_0_106262:($
output_0_106264:$!
output_1_106279:($
output_1_106281:$!
output_2_106296:($
output_2_106298:$!
output_3_106313:($
output_3_106315:$!
output_4_106330:($
output_4_106332:$
identity??01st_fully_connected_of_0/StatefulPartitionedCall?01st_fully_connected_of_1/StatefulPartitionedCall?01st_fully_connected_of_2/StatefulPartitionedCall?01st_fully_connected_of_3/StatefulPartitionedCall?01st_fully_connected_of_4/StatefulPartitionedCall? 4th_conv/StatefulPartitionedCall? 5th_conv/StatefulPartitionedCall? 6th_conv/StatefulPartitionedCall?-batch_normalization_3/StatefulPartitionedCall?-batch_normalization_4/StatefulPartitionedCall?-batch_normalization_5/StatefulPartitionedCall? output_0/StatefulPartitionedCall? output_1/StatefulPartitionedCall? output_2/StatefulPartitionedCall? output_3/StatefulPartitionedCall? output_4/StatefulPartitionedCall?
 4th_conv/StatefulPartitionedCallStatefulPartitionedCallinputsth_conv_106053th_conv_106055*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????dd *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_4th_conv_layer_call_and_return_conditional_losses_106052?
4th_pool/PartitionedCallPartitionedCall)4th_conv/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????22 * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_4th_pool_layer_call_and_return_conditional_losses_105815?
-batch_normalization_3/StatefulPartitionedCallStatefulPartitionedCall!4th_pool/PartitionedCall:output:0batch_normalization_3_106059batch_normalization_3_106061batch_normalization_3_106063batch_normalization_3_106065*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????22 *&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Z
fURS
Q__inference_batch_normalization_3_layer_call_and_return_conditional_losses_105840?
 5th_conv/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_3/StatefulPartitionedCall:output:0th_conv_106080th_conv_106082*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????22 *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_5th_conv_layer_call_and_return_conditional_losses_106079?
5th_pool/PartitionedCallPartitionedCall)5th_conv/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_5th_pool_layer_call_and_return_conditional_losses_105891?
-batch_normalization_4/StatefulPartitionedCallStatefulPartitionedCall!5th_pool/PartitionedCall:output:0batch_normalization_4_106086batch_normalization_4_106088batch_normalization_4_106090batch_normalization_4_106092*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? *&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Z
fURS
Q__inference_batch_normalization_4_layer_call_and_return_conditional_losses_105916?
 6th_conv/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_4/StatefulPartitionedCall:output:0th_conv_106107th_conv_106109*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_6th_conv_layer_call_and_return_conditional_losses_106106?
6th_pool/PartitionedCallPartitionedCall)6th_conv/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_6th_pool_layer_call_and_return_conditional_losses_105967?
-batch_normalization_5/StatefulPartitionedCallStatefulPartitionedCall!6th_pool/PartitionedCall:output:0batch_normalization_5_106113batch_normalization_5_106115batch_normalization_5_106117batch_normalization_5_106119*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? *&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Z
fURS
Q__inference_batch_normalization_5_layer_call_and_return_conditional_losses_105992?
flat/PartitionedCallPartitionedCall6batch_normalization_5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????$* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *I
fDRB
@__inference_flat_layer_call_and_return_conditional_losses_106128?
01st_fully_connected_of_4/StatefulPartitionedCallStatefulPartitionedCallflat/PartitionedCall:output:0st_fully_connected_of_4_106142st_fully_connected_of_4_106144*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????(*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *]
fXRV
T__inference_1st_fully_connected_of_4_layer_call_and_return_conditional_losses_106141?
01st_fully_connected_of_3/StatefulPartitionedCallStatefulPartitionedCallflat/PartitionedCall:output:0st_fully_connected_of_3_106159st_fully_connected_of_3_106161*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????(*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *]
fXRV
T__inference_1st_fully_connected_of_3_layer_call_and_return_conditional_losses_106158?
01st_fully_connected_of_2/StatefulPartitionedCallStatefulPartitionedCallflat/PartitionedCall:output:0st_fully_connected_of_2_106176st_fully_connected_of_2_106178*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????(*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *]
fXRV
T__inference_1st_fully_connected_of_2_layer_call_and_return_conditional_losses_106175?
01st_fully_connected_of_1/StatefulPartitionedCallStatefulPartitionedCallflat/PartitionedCall:output:0st_fully_connected_of_1_106193st_fully_connected_of_1_106195*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????(*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *]
fXRV
T__inference_1st_fully_connected_of_1_layer_call_and_return_conditional_losses_106192?
01st_fully_connected_of_0/StatefulPartitionedCallStatefulPartitionedCallflat/PartitionedCall:output:0st_fully_connected_of_0_106210st_fully_connected_of_0_106212*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????(*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *]
fXRV
T__inference_1st_fully_connected_of_0_layer_call_and_return_conditional_losses_106209?
dropout_4/PartitionedCallPartitionedCall91st_fully_connected_of_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????(* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dropout_4_layer_call_and_return_conditional_losses_106220?
dropout_3/PartitionedCallPartitionedCall91st_fully_connected_of_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????(* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dropout_3_layer_call_and_return_conditional_losses_106227?
dropout_2/PartitionedCallPartitionedCall91st_fully_connected_of_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????(* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dropout_2_layer_call_and_return_conditional_losses_106234?
dropout_1/PartitionedCallPartitionedCall91st_fully_connected_of_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????(* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dropout_1_layer_call_and_return_conditional_losses_106241?
dropout/PartitionedCallPartitionedCall91st_fully_connected_of_0/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????(* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dropout_layer_call_and_return_conditional_losses_106248?
 output_0/StatefulPartitionedCallStatefulPartitionedCall dropout/PartitionedCall:output:0output_0_106262output_0_106264*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????$*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_output_0_layer_call_and_return_conditional_losses_106261?
 output_1/StatefulPartitionedCallStatefulPartitionedCall"dropout_1/PartitionedCall:output:0output_1_106279output_1_106281*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????$*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_output_1_layer_call_and_return_conditional_losses_106278?
 output_2/StatefulPartitionedCallStatefulPartitionedCall"dropout_2/PartitionedCall:output:0output_2_106296output_2_106298*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????$*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_output_2_layer_call_and_return_conditional_losses_106295?
 output_3/StatefulPartitionedCallStatefulPartitionedCall"dropout_3/PartitionedCall:output:0output_3_106313output_3_106315*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????$*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_output_3_layer_call_and_return_conditional_losses_106312?
 output_4/StatefulPartitionedCallStatefulPartitionedCall"dropout_4/PartitionedCall:output:0output_4_106330output_4_106332*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????$*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_output_4_layer_call_and_return_conditional_losses_106329?
tf.convert_to_tensor_1/CastCast)output_0/StatefulPartitionedCall:output:0*

DstT0*

SrcT0*'
_output_shapes
:?????????$?
tf.convert_to_tensor_1/Cast_1Cast)output_1/StatefulPartitionedCall:output:0*

DstT0*

SrcT0*'
_output_shapes
:?????????$?
tf.convert_to_tensor_1/Cast_2Cast)output_2/StatefulPartitionedCall:output:0*

DstT0*

SrcT0*'
_output_shapes
:?????????$?
tf.convert_to_tensor_1/Cast_3Cast)output_3/StatefulPartitionedCall:output:0*

DstT0*

SrcT0*'
_output_shapes
:?????????$?
tf.convert_to_tensor_1/Cast_4Cast)output_4/StatefulPartitionedCall:output:0*

DstT0*

SrcT0*'
_output_shapes
:?????????$?
tf.convert_to_tensor_1/packedPacktf.convert_to_tensor_1/Cast:y:0!tf.convert_to_tensor_1/Cast_1:y:0!tf.convert_to_tensor_1/Cast_2:y:0!tf.convert_to_tensor_1/Cast_3:y:0!tf.convert_to_tensor_1/Cast_4:y:0*
N*
T0*+
_output_shapes
:?????????$|
'tf.compat.v1.transpose_1/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          ?
"tf.compat.v1.transpose_1/transpose	Transpose&tf.convert_to_tensor_1/packed:output:00tf.compat.v1.transpose_1/transpose/perm:output:0*
T0*+
_output_shapes
:?????????$y
IdentityIdentity&tf.compat.v1.transpose_1/transpose:y:0^NoOp*
T0*+
_output_shapes
:?????????$?
NoOpNoOp1^1st_fully_connected_of_0/StatefulPartitionedCall1^1st_fully_connected_of_1/StatefulPartitionedCall1^1st_fully_connected_of_2/StatefulPartitionedCall1^1st_fully_connected_of_3/StatefulPartitionedCall1^1st_fully_connected_of_4/StatefulPartitionedCall!^4th_conv/StatefulPartitionedCall!^5th_conv/StatefulPartitionedCall!^6th_conv/StatefulPartitionedCall.^batch_normalization_3/StatefulPartitionedCall.^batch_normalization_4/StatefulPartitionedCall.^batch_normalization_5/StatefulPartitionedCall!^output_0/StatefulPartitionedCall!^output_1/StatefulPartitionedCall!^output_2/StatefulPartitionedCall!^output_3/StatefulPartitionedCall!^output_4/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*z
_input_shapesi
g:?????????dd: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2d
01st_fully_connected_of_0/StatefulPartitionedCall01st_fully_connected_of_0/StatefulPartitionedCall2d
01st_fully_connected_of_1/StatefulPartitionedCall01st_fully_connected_of_1/StatefulPartitionedCall2d
01st_fully_connected_of_2/StatefulPartitionedCall01st_fully_connected_of_2/StatefulPartitionedCall2d
01st_fully_connected_of_3/StatefulPartitionedCall01st_fully_connected_of_3/StatefulPartitionedCall2d
01st_fully_connected_of_4/StatefulPartitionedCall01st_fully_connected_of_4/StatefulPartitionedCall2D
 4th_conv/StatefulPartitionedCall 4th_conv/StatefulPartitionedCall2D
 5th_conv/StatefulPartitionedCall 5th_conv/StatefulPartitionedCall2D
 6th_conv/StatefulPartitionedCall 6th_conv/StatefulPartitionedCall2^
-batch_normalization_3/StatefulPartitionedCall-batch_normalization_3/StatefulPartitionedCall2^
-batch_normalization_4/StatefulPartitionedCall-batch_normalization_4/StatefulPartitionedCall2^
-batch_normalization_5/StatefulPartitionedCall-batch_normalization_5/StatefulPartitionedCall2D
 output_0/StatefulPartitionedCall output_0/StatefulPartitionedCall2D
 output_1/StatefulPartitionedCall output_1/StatefulPartitionedCall2D
 output_2/StatefulPartitionedCall output_2/StatefulPartitionedCall2D
 output_3/StatefulPartitionedCall output_3/StatefulPartitionedCall2D
 output_4/StatefulPartitionedCall output_4/StatefulPartitionedCall:W S
/
_output_shapes
:?????????dd
 
_user_specified_nameinputs
?
F
*__inference_dropout_3_layer_call_fn_108325

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????(* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dropout_3_layer_call_and_return_conditional_losses_106227`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:?????????("
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:?????????(:O K
'
_output_shapes
:?????????(
 
_user_specified_nameinputs
?
F
*__inference_dropout_2_layer_call_fn_108298

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????(* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dropout_2_layer_call_and_return_conditional_losses_106234`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:?????????("
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:?????????(:O K
'
_output_shapes
:?????????(
 
_user_specified_nameinputs
?

?
D__inference_output_2_layer_call_and_return_conditional_losses_106295

inputs0
matmul_readvariableop_resource:($-
biasadd_readvariableop_resource:$
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:($*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????$r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:$*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????$V
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:?????????$`
IdentityIdentitySoftmax:softmax:0^NoOp*
T0*'
_output_shapes
:?????????$w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????(: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????(
 
_user_specified_nameinputs
?	
?
6__inference_batch_normalization_5_layer_call_fn_108079

inputs
unknown: 
	unknown_0: 
	unknown_1: 
	unknown_2: 
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+??????????????????????????? *&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Z
fURS
Q__inference_batch_normalization_5_layer_call_and_return_conditional_losses_105992?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+??????????????????????????? `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+??????????????????????????? : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs
?
c
E__inference_dropout_3_layer_call_and_return_conditional_losses_106227

inputs

identity_1N
IdentityIdentityinputs*
T0*'
_output_shapes
:?????????([

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:?????????("!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:?????????(:O K
'
_output_shapes
:?????????(
 
_user_specified_nameinputs
?
?
)__inference_output_3_layer_call_fn_108443

inputs
unknown:($
	unknown_0:$
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????$*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_output_3_layer_call_and_return_conditional_losses_106312o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????$`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????(: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????(
 
_user_specified_nameinputs
?
?
9__inference_1st_fully_connected_of_3_layer_call_fn_108208

inputs
unknown:	?$(
	unknown_0:(
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????(*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *]
fXRV
T__inference_1st_fully_connected_of_3_layer_call_and_return_conditional_losses_106158o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????(`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????$: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????$
 
_user_specified_nameinputs
?
F
*__inference_dropout_4_layer_call_fn_108352

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????(* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dropout_4_layer_call_and_return_conditional_losses_106220`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:?????????("
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:?????????(:O K
'
_output_shapes
:?????????(
 
_user_specified_nameinputs
?
?
Q__inference_batch_normalization_4_layer_call_and_return_conditional_losses_105947

inputs%
readvariableop_resource: '
readvariableop_1_resource: 6
(fusedbatchnormv3_readvariableop_resource: 8
*fusedbatchnormv3_readvariableop_1_resource: 
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype0?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+??????????????????????????? : : : : :*
epsilon%o?:*
exponential_avg_factor%
?#<?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(}
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+??????????????????????????? ?
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+??????????????????????????? : : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs
Ԉ
?
A__inference_model_layer_call_and_return_conditional_losses_106870

inputs(
th_conv_106760: 
th_conv_106762: *
batch_normalization_3_106766: *
batch_normalization_3_106768: *
batch_normalization_3_106770: *
batch_normalization_3_106772: (
th_conv_106775:  
th_conv_106777: *
batch_normalization_4_106781: *
batch_normalization_4_106783: *
batch_normalization_4_106785: *
batch_normalization_4_106787: (
th_conv_106790:  
th_conv_106792: *
batch_normalization_5_106796: *
batch_normalization_5_106798: *
batch_normalization_5_106800: *
batch_normalization_5_106802: 1
st_fully_connected_of_4_106806:	?$(,
st_fully_connected_of_4_106808:(1
st_fully_connected_of_3_106811:	?$(,
st_fully_connected_of_3_106813:(1
st_fully_connected_of_2_106816:	?$(,
st_fully_connected_of_2_106818:(1
st_fully_connected_of_1_106821:	?$(,
st_fully_connected_of_1_106823:(1
st_fully_connected_of_0_106826:	?$(,
st_fully_connected_of_0_106828:(!
output_0_106836:($
output_0_106838:$!
output_1_106841:($
output_1_106843:$!
output_2_106846:($
output_2_106848:$!
output_3_106851:($
output_3_106853:$!
output_4_106856:($
output_4_106858:$
identity??01st_fully_connected_of_0/StatefulPartitionedCall?01st_fully_connected_of_1/StatefulPartitionedCall?01st_fully_connected_of_2/StatefulPartitionedCall?01st_fully_connected_of_3/StatefulPartitionedCall?01st_fully_connected_of_4/StatefulPartitionedCall? 4th_conv/StatefulPartitionedCall? 5th_conv/StatefulPartitionedCall? 6th_conv/StatefulPartitionedCall?-batch_normalization_3/StatefulPartitionedCall?-batch_normalization_4/StatefulPartitionedCall?-batch_normalization_5/StatefulPartitionedCall?dropout/StatefulPartitionedCall?!dropout_1/StatefulPartitionedCall?!dropout_2/StatefulPartitionedCall?!dropout_3/StatefulPartitionedCall?!dropout_4/StatefulPartitionedCall? output_0/StatefulPartitionedCall? output_1/StatefulPartitionedCall? output_2/StatefulPartitionedCall? output_3/StatefulPartitionedCall? output_4/StatefulPartitionedCall?
 4th_conv/StatefulPartitionedCallStatefulPartitionedCallinputsth_conv_106760th_conv_106762*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????dd *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_4th_conv_layer_call_and_return_conditional_losses_106052?
4th_pool/PartitionedCallPartitionedCall)4th_conv/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????22 * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_4th_pool_layer_call_and_return_conditional_losses_105815?
-batch_normalization_3/StatefulPartitionedCallStatefulPartitionedCall!4th_pool/PartitionedCall:output:0batch_normalization_3_106766batch_normalization_3_106768batch_normalization_3_106770batch_normalization_3_106772*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????22 *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Z
fURS
Q__inference_batch_normalization_3_layer_call_and_return_conditional_losses_105871?
 5th_conv/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_3/StatefulPartitionedCall:output:0th_conv_106775th_conv_106777*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????22 *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_5th_conv_layer_call_and_return_conditional_losses_106079?
5th_pool/PartitionedCallPartitionedCall)5th_conv/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_5th_pool_layer_call_and_return_conditional_losses_105891?
-batch_normalization_4/StatefulPartitionedCallStatefulPartitionedCall!5th_pool/PartitionedCall:output:0batch_normalization_4_106781batch_normalization_4_106783batch_normalization_4_106785batch_normalization_4_106787*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Z
fURS
Q__inference_batch_normalization_4_layer_call_and_return_conditional_losses_105947?
 6th_conv/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_4/StatefulPartitionedCall:output:0th_conv_106790th_conv_106792*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_6th_conv_layer_call_and_return_conditional_losses_106106?
6th_pool/PartitionedCallPartitionedCall)6th_conv/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_6th_pool_layer_call_and_return_conditional_losses_105967?
-batch_normalization_5/StatefulPartitionedCallStatefulPartitionedCall!6th_pool/PartitionedCall:output:0batch_normalization_5_106796batch_normalization_5_106798batch_normalization_5_106800batch_normalization_5_106802*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Z
fURS
Q__inference_batch_normalization_5_layer_call_and_return_conditional_losses_106023?
flat/PartitionedCallPartitionedCall6batch_normalization_5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????$* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *I
fDRB
@__inference_flat_layer_call_and_return_conditional_losses_106128?
01st_fully_connected_of_4/StatefulPartitionedCallStatefulPartitionedCallflat/PartitionedCall:output:0st_fully_connected_of_4_106806st_fully_connected_of_4_106808*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????(*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *]
fXRV
T__inference_1st_fully_connected_of_4_layer_call_and_return_conditional_losses_106141?
01st_fully_connected_of_3/StatefulPartitionedCallStatefulPartitionedCallflat/PartitionedCall:output:0st_fully_connected_of_3_106811st_fully_connected_of_3_106813*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????(*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *]
fXRV
T__inference_1st_fully_connected_of_3_layer_call_and_return_conditional_losses_106158?
01st_fully_connected_of_2/StatefulPartitionedCallStatefulPartitionedCallflat/PartitionedCall:output:0st_fully_connected_of_2_106816st_fully_connected_of_2_106818*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????(*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *]
fXRV
T__inference_1st_fully_connected_of_2_layer_call_and_return_conditional_losses_106175?
01st_fully_connected_of_1/StatefulPartitionedCallStatefulPartitionedCallflat/PartitionedCall:output:0st_fully_connected_of_1_106821st_fully_connected_of_1_106823*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????(*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *]
fXRV
T__inference_1st_fully_connected_of_1_layer_call_and_return_conditional_losses_106192?
01st_fully_connected_of_0/StatefulPartitionedCallStatefulPartitionedCallflat/PartitionedCall:output:0st_fully_connected_of_0_106826st_fully_connected_of_0_106828*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????(*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *]
fXRV
T__inference_1st_fully_connected_of_0_layer_call_and_return_conditional_losses_106209?
!dropout_4/StatefulPartitionedCallStatefulPartitionedCall91st_fully_connected_of_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????(* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dropout_4_layer_call_and_return_conditional_losses_106585?
!dropout_3/StatefulPartitionedCallStatefulPartitionedCall91st_fully_connected_of_3/StatefulPartitionedCall:output:0"^dropout_4/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????(* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dropout_3_layer_call_and_return_conditional_losses_106562?
!dropout_2/StatefulPartitionedCallStatefulPartitionedCall91st_fully_connected_of_2/StatefulPartitionedCall:output:0"^dropout_3/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????(* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dropout_2_layer_call_and_return_conditional_losses_106539?
!dropout_1/StatefulPartitionedCallStatefulPartitionedCall91st_fully_connected_of_1/StatefulPartitionedCall:output:0"^dropout_2/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????(* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dropout_1_layer_call_and_return_conditional_losses_106516?
dropout/StatefulPartitionedCallStatefulPartitionedCall91st_fully_connected_of_0/StatefulPartitionedCall:output:0"^dropout_1/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????(* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dropout_layer_call_and_return_conditional_losses_106493?
 output_0/StatefulPartitionedCallStatefulPartitionedCall(dropout/StatefulPartitionedCall:output:0output_0_106836output_0_106838*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????$*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_output_0_layer_call_and_return_conditional_losses_106261?
 output_1/StatefulPartitionedCallStatefulPartitionedCall*dropout_1/StatefulPartitionedCall:output:0output_1_106841output_1_106843*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????$*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_output_1_layer_call_and_return_conditional_losses_106278?
 output_2/StatefulPartitionedCallStatefulPartitionedCall*dropout_2/StatefulPartitionedCall:output:0output_2_106846output_2_106848*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????$*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_output_2_layer_call_and_return_conditional_losses_106295?
 output_3/StatefulPartitionedCallStatefulPartitionedCall*dropout_3/StatefulPartitionedCall:output:0output_3_106851output_3_106853*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????$*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_output_3_layer_call_and_return_conditional_losses_106312?
 output_4/StatefulPartitionedCallStatefulPartitionedCall*dropout_4/StatefulPartitionedCall:output:0output_4_106856output_4_106858*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????$*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_output_4_layer_call_and_return_conditional_losses_106329?
tf.convert_to_tensor_1/CastCast)output_0/StatefulPartitionedCall:output:0*

DstT0*

SrcT0*'
_output_shapes
:?????????$?
tf.convert_to_tensor_1/Cast_1Cast)output_1/StatefulPartitionedCall:output:0*

DstT0*

SrcT0*'
_output_shapes
:?????????$?
tf.convert_to_tensor_1/Cast_2Cast)output_2/StatefulPartitionedCall:output:0*

DstT0*

SrcT0*'
_output_shapes
:?????????$?
tf.convert_to_tensor_1/Cast_3Cast)output_3/StatefulPartitionedCall:output:0*

DstT0*

SrcT0*'
_output_shapes
:?????????$?
tf.convert_to_tensor_1/Cast_4Cast)output_4/StatefulPartitionedCall:output:0*

DstT0*

SrcT0*'
_output_shapes
:?????????$?
tf.convert_to_tensor_1/packedPacktf.convert_to_tensor_1/Cast:y:0!tf.convert_to_tensor_1/Cast_1:y:0!tf.convert_to_tensor_1/Cast_2:y:0!tf.convert_to_tensor_1/Cast_3:y:0!tf.convert_to_tensor_1/Cast_4:y:0*
N*
T0*+
_output_shapes
:?????????$|
'tf.compat.v1.transpose_1/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          ?
"tf.compat.v1.transpose_1/transpose	Transpose&tf.convert_to_tensor_1/packed:output:00tf.compat.v1.transpose_1/transpose/perm:output:0*
T0*+
_output_shapes
:?????????$y
IdentityIdentity&tf.compat.v1.transpose_1/transpose:y:0^NoOp*
T0*+
_output_shapes
:?????????$?
NoOpNoOp1^1st_fully_connected_of_0/StatefulPartitionedCall1^1st_fully_connected_of_1/StatefulPartitionedCall1^1st_fully_connected_of_2/StatefulPartitionedCall1^1st_fully_connected_of_3/StatefulPartitionedCall1^1st_fully_connected_of_4/StatefulPartitionedCall!^4th_conv/StatefulPartitionedCall!^5th_conv/StatefulPartitionedCall!^6th_conv/StatefulPartitionedCall.^batch_normalization_3/StatefulPartitionedCall.^batch_normalization_4/StatefulPartitionedCall.^batch_normalization_5/StatefulPartitionedCall ^dropout/StatefulPartitionedCall"^dropout_1/StatefulPartitionedCall"^dropout_2/StatefulPartitionedCall"^dropout_3/StatefulPartitionedCall"^dropout_4/StatefulPartitionedCall!^output_0/StatefulPartitionedCall!^output_1/StatefulPartitionedCall!^output_2/StatefulPartitionedCall!^output_3/StatefulPartitionedCall!^output_4/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*z
_input_shapesi
g:?????????dd: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2d
01st_fully_connected_of_0/StatefulPartitionedCall01st_fully_connected_of_0/StatefulPartitionedCall2d
01st_fully_connected_of_1/StatefulPartitionedCall01st_fully_connected_of_1/StatefulPartitionedCall2d
01st_fully_connected_of_2/StatefulPartitionedCall01st_fully_connected_of_2/StatefulPartitionedCall2d
01st_fully_connected_of_3/StatefulPartitionedCall01st_fully_connected_of_3/StatefulPartitionedCall2d
01st_fully_connected_of_4/StatefulPartitionedCall01st_fully_connected_of_4/StatefulPartitionedCall2D
 4th_conv/StatefulPartitionedCall 4th_conv/StatefulPartitionedCall2D
 5th_conv/StatefulPartitionedCall 5th_conv/StatefulPartitionedCall2D
 6th_conv/StatefulPartitionedCall 6th_conv/StatefulPartitionedCall2^
-batch_normalization_3/StatefulPartitionedCall-batch_normalization_3/StatefulPartitionedCall2^
-batch_normalization_4/StatefulPartitionedCall-batch_normalization_4/StatefulPartitionedCall2^
-batch_normalization_5/StatefulPartitionedCall-batch_normalization_5/StatefulPartitionedCall2B
dropout/StatefulPartitionedCalldropout/StatefulPartitionedCall2F
!dropout_1/StatefulPartitionedCall!dropout_1/StatefulPartitionedCall2F
!dropout_2/StatefulPartitionedCall!dropout_2/StatefulPartitionedCall2F
!dropout_3/StatefulPartitionedCall!dropout_3/StatefulPartitionedCall2F
!dropout_4/StatefulPartitionedCall!dropout_4/StatefulPartitionedCall2D
 output_0/StatefulPartitionedCall output_0/StatefulPartitionedCall2D
 output_1/StatefulPartitionedCall output_1/StatefulPartitionedCall2D
 output_2/StatefulPartitionedCall output_2/StatefulPartitionedCall2D
 output_3/StatefulPartitionedCall output_3/StatefulPartitionedCall2D
 output_4/StatefulPartitionedCall output_4/StatefulPartitionedCall:W S
/
_output_shapes
:?????????dd
 
_user_specified_nameinputs
?
?
9__inference_1st_fully_connected_of_0_layer_call_fn_108148

inputs
unknown:	?$(
	unknown_0:(
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????(*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *]
fXRV
T__inference_1st_fully_connected_of_0_layer_call_and_return_conditional_losses_106209o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????(`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????$: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????$
 
_user_specified_nameinputs
?
\
@__inference_flat_layer_call_and_return_conditional_losses_108139

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"????   ]
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:??????????$Y
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:??????????$"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:????????? :W S
/
_output_shapes
:????????? 
 
_user_specified_nameinputs
?

?
D__inference_output_0_layer_call_and_return_conditional_losses_108394

inputs0
matmul_readvariableop_resource:($-
biasadd_readvariableop_resource:$
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:($*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????$r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:$*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????$V
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:?????????$`
IdentityIdentitySoftmax:softmax:0^NoOp*
T0*'
_output_shapes
:?????????$w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????(: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????(
 
_user_specified_nameinputs
?
?
Q__inference_batch_normalization_3_layer_call_and_return_conditional_losses_105840

inputs%
readvariableop_resource: '
readvariableop_1_resource: 6
(fusedbatchnormv3_readvariableop_resource: 8
*fusedbatchnormv3_readvariableop_1_resource: 
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype0?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+??????????????????????????? : : : : :*
epsilon%o?:*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+??????????????????????????? ?
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+??????????????????????????? : : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs
?

?
T__inference_1st_fully_connected_of_2_layer_call_and_return_conditional_losses_108199

inputs1
matmul_readvariableop_resource:	?$(-
biasadd_readvariableop_resource:(
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?$(*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????(r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:(*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????(P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:?????????(a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:?????????(w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????$: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????$
 
_user_specified_nameinputs
?
a
C__inference_dropout_layer_call_and_return_conditional_losses_108254

inputs

identity_1N
IdentityIdentityinputs*
T0*'
_output_shapes
:?????????([

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:?????????("!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:?????????(:O K
'
_output_shapes
:?????????(
 
_user_specified_nameinputs
?
c
*__inference_dropout_3_layer_call_fn_108330

inputs
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????(* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dropout_3_layer_call_and_return_conditional_losses_106562o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????(`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:?????????(22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????(
 
_user_specified_nameinputs
?	
d
E__inference_dropout_4_layer_call_and_return_conditional_losses_106585

inputs
identity?R
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??d
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:?????????(C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:?????????(*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????(o
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????(i
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:?????????(Y
IdentityIdentitydropout/Mul_1:z:0*
T0*'
_output_shapes
:?????????("
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:?????????(:O K
'
_output_shapes
:?????????(
 
_user_specified_nameinputs
?
E
)__inference_5th_pool_layer_call_fn_107969

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4????????????????????????????????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_5th_pool_layer_call_and_return_conditional_losses_105891?
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
`
D__inference_4th_pool_layer_call_and_return_conditional_losses_105815

inputs
identity?
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4????????????????????????????????????*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
`
D__inference_6th_pool_layer_call_and_return_conditional_losses_108066

inputs
identity?
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4????????????????????????????????????*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?

?
D__inference_output_4_layer_call_and_return_conditional_losses_106329

inputs0
matmul_readvariableop_resource:($-
biasadd_readvariableop_resource:$
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:($*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????$r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:$*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????$V
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:?????????$`
IdentityIdentitySoftmax:softmax:0^NoOp*
T0*'
_output_shapes
:?????????$w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????(: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????(
 
_user_specified_nameinputs
?

?
T__inference_1st_fully_connected_of_3_layer_call_and_return_conditional_losses_108219

inputs1
matmul_readvariableop_resource:	?$(-
biasadd_readvariableop_resource:(
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?$(*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????(r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:(*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????(P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:?????????(a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:?????????(w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????$: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????$
 
_user_specified_nameinputs
?
?
)__inference_output_4_layer_call_fn_108463

inputs
unknown:($
	unknown_0:$
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????$*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_output_4_layer_call_and_return_conditional_losses_106329o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????$`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????(: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????(
 
_user_specified_nameinputs
?
`
D__inference_6th_pool_layer_call_and_return_conditional_losses_105967

inputs
identity?
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4????????????????????????????????????*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
?
)__inference_output_1_layer_call_fn_108403

inputs
unknown:($
	unknown_0:$
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????$*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_output_1_layer_call_and_return_conditional_losses_106278o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????$`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????(: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????(
 
_user_specified_nameinputs
?	
b
C__inference_dropout_layer_call_and_return_conditional_losses_108266

inputs
identity?R
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??d
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:?????????(C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:?????????(*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????(o
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????(i
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:?????????(Y
IdentityIdentitydropout/Mul_1:z:0*
T0*'
_output_shapes
:?????????("
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:?????????(:O K
'
_output_shapes
:?????????(
 
_user_specified_nameinputs
?
?
&__inference_model_layer_call_fn_107507

inputs!
unknown: 
	unknown_0: 
	unknown_1: 
	unknown_2: 
	unknown_3: 
	unknown_4: #
	unknown_5:  
	unknown_6: 
	unknown_7: 
	unknown_8: 
	unknown_9: 

unknown_10: $

unknown_11:  

unknown_12: 

unknown_13: 

unknown_14: 

unknown_15: 

unknown_16: 

unknown_17:	?$(

unknown_18:(

unknown_19:	?$(

unknown_20:(

unknown_21:	?$(

unknown_22:(

unknown_23:	?$(

unknown_24:(

unknown_25:	?$(

unknown_26:(

unknown_27:($

unknown_28:$

unknown_29:($

unknown_30:$

unknown_31:($

unknown_32:$

unknown_33:($

unknown_34:$

unknown_35:($

unknown_36:$
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36*2
Tin+
)2'*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????$*B
_read_only_resource_inputs$
" 	
 !"#$%&*-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_model_layer_call_and_return_conditional_losses_106870s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:?????????$`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*z
_input_shapesi
g:?????????dd: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????dd
 
_user_specified_nameinputs
?
c
E__inference_dropout_1_layer_call_and_return_conditional_losses_106241

inputs

identity_1N
IdentityIdentityinputs*
T0*'
_output_shapes
:?????????([

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:?????????("!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:?????????(:O K
'
_output_shapes
:?????????(
 
_user_specified_nameinputs
?

?
D__inference_output_1_layer_call_and_return_conditional_losses_108414

inputs0
matmul_readvariableop_resource:($-
biasadd_readvariableop_resource:$
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:($*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????$r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:$*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????$V
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:?????????$`
IdentityIdentitySoftmax:softmax:0^NoOp*
T0*'
_output_shapes
:?????????$w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????(: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????(
 
_user_specified_nameinputs
?
c
E__inference_dropout_4_layer_call_and_return_conditional_losses_108362

inputs

identity_1N
IdentityIdentityinputs*
T0*'
_output_shapes
:?????????([

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:?????????("!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:?????????(:O K
'
_output_shapes
:?????????(
 
_user_specified_nameinputs
?

?
T__inference_1st_fully_connected_of_0_layer_call_and_return_conditional_losses_106209

inputs1
matmul_readvariableop_resource:	?$(-
biasadd_readvariableop_resource:(
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?$(*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????(r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:(*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????(P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:?????????(a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:?????????(w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????$: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????$
 
_user_specified_nameinputs
?
?
D__inference_4th_conv_layer_call_and_return_conditional_losses_107872

inputs8
conv2d_readvariableop_resource: -
biasadd_readvariableop_resource: 
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype0?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????dd *
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????dd X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:?????????dd i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:?????????dd w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????dd: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????dd
 
_user_specified_nameinputs
?
c
E__inference_dropout_3_layer_call_and_return_conditional_losses_108335

inputs

identity_1N
IdentityIdentityinputs*
T0*'
_output_shapes
:?????????([

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:?????????("!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:?????????(:O K
'
_output_shapes
:?????????(
 
_user_specified_nameinputs
?
?
Q__inference_batch_normalization_4_layer_call_and_return_conditional_losses_108018

inputs%
readvariableop_resource: '
readvariableop_1_resource: 6
(fusedbatchnormv3_readvariableop_resource: 8
*fusedbatchnormv3_readvariableop_1_resource: 
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype0?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+??????????????????????????? : : : : :*
epsilon%o?:*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+??????????????????????????? ?
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+??????????????????????????? : : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs
?
c
E__inference_dropout_2_layer_call_and_return_conditional_losses_106234

inputs

identity_1N
IdentityIdentityinputs*
T0*'
_output_shapes
:?????????([

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:?????????("!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:?????????(:O K
'
_output_shapes
:?????????(
 
_user_specified_nameinputs
?
D
(__inference_dropout_layer_call_fn_108244

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????(* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dropout_layer_call_and_return_conditional_losses_106248`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:?????????("
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:?????????(:O K
'
_output_shapes
:?????????(
 
_user_specified_nameinputs
?

?
T__inference_1st_fully_connected_of_2_layer_call_and_return_conditional_losses_106175

inputs1
matmul_readvariableop_resource:	?$(-
biasadd_readvariableop_resource:(
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?$(*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????(r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:(*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????(P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:?????????(a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:?????????(w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????$: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????$
 
_user_specified_nameinputs
?
?
9__inference_1st_fully_connected_of_2_layer_call_fn_108188

inputs
unknown:	?$(
	unknown_0:(
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????(*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *]
fXRV
T__inference_1st_fully_connected_of_2_layer_call_and_return_conditional_losses_106175o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????(`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????$: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????$
 
_user_specified_nameinputs
?

?
D__inference_output_2_layer_call_and_return_conditional_losses_108434

inputs0
matmul_readvariableop_resource:($-
biasadd_readvariableop_resource:$
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:($*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????$r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:$*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????$V
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:?????????$`
IdentityIdentitySoftmax:softmax:0^NoOp*
T0*'
_output_shapes
:?????????$w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????(: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????(
 
_user_specified_nameinputs
?
?
)__inference_output_2_layer_call_fn_108423

inputs
unknown:($
	unknown_0:$
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????$*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_output_2_layer_call_and_return_conditional_losses_106295o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????$`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????(: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????(
 
_user_specified_nameinputs
?
F
*__inference_dropout_1_layer_call_fn_108271

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????(* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dropout_1_layer_call_and_return_conditional_losses_106241`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:?????????("
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:?????????(:O K
'
_output_shapes
:?????????(
 
_user_specified_nameinputs
?
?
Q__inference_batch_normalization_4_layer_call_and_return_conditional_losses_108036

inputs%
readvariableop_resource: '
readvariableop_1_resource: 6
(fusedbatchnormv3_readvariableop_resource: 8
*fusedbatchnormv3_readvariableop_1_resource: 
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype0?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+??????????????????????????? : : : : :*
epsilon%o?:*
exponential_avg_factor%
?#<?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(}
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+??????????????????????????? ?
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+??????????????????????????? : : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs
?	
d
E__inference_dropout_1_layer_call_and_return_conditional_losses_108293

inputs
identity?R
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??d
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:?????????(C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:?????????(*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????(o
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????(i
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:?????????(Y
IdentityIdentitydropout/Mul_1:z:0*
T0*'
_output_shapes
:?????????("
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:?????????(:O K
'
_output_shapes
:?????????(
 
_user_specified_nameinputs
?
?
D__inference_6th_conv_layer_call_and_return_conditional_losses_108056

inputs8
conv2d_readvariableop_resource:  -
biasadd_readvariableop_resource: 
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:  *
dtype0?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? *
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:????????? i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:????????? w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:????????? : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:????????? 
 
_user_specified_nameinputs
?

?
T__inference_1st_fully_connected_of_0_layer_call_and_return_conditional_losses_108159

inputs1
matmul_readvariableop_resource:	?$(-
biasadd_readvariableop_resource:(
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?$(*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????(r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:(*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????(P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:?????????(a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:?????????(w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????$: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????$
 
_user_specified_nameinputs
?
?
Q__inference_batch_normalization_5_layer_call_and_return_conditional_losses_106023

inputs%
readvariableop_resource: '
readvariableop_1_resource: 6
(fusedbatchnormv3_readvariableop_resource: 8
*fusedbatchnormv3_readvariableop_1_resource: 
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype0?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+??????????????????????????? : : : : :*
epsilon%o?:*
exponential_avg_factor%
?#<?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(}
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+??????????????????????????? ?
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+??????????????????????????? : : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs
?
E
)__inference_6th_pool_layer_call_fn_108061

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4????????????????????????????????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_6th_pool_layer_call_and_return_conditional_losses_105967?
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?	
?
6__inference_batch_normalization_4_layer_call_fn_107987

inputs
unknown: 
	unknown_0: 
	unknown_1: 
	unknown_2: 
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+??????????????????????????? *&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Z
fURS
Q__inference_batch_normalization_4_layer_call_and_return_conditional_losses_105916?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+??????????????????????????? `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+??????????????????????????? : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs
?
?
&__inference_model_layer_call_fn_106423

images!
unknown: 
	unknown_0: 
	unknown_1: 
	unknown_2: 
	unknown_3: 
	unknown_4: #
	unknown_5:  
	unknown_6: 
	unknown_7: 
	unknown_8: 
	unknown_9: 

unknown_10: $

unknown_11:  

unknown_12: 

unknown_13: 

unknown_14: 

unknown_15: 

unknown_16: 

unknown_17:	?$(

unknown_18:(

unknown_19:	?$(

unknown_20:(

unknown_21:	?$(

unknown_22:(

unknown_23:	?$(

unknown_24:(

unknown_25:	?$(

unknown_26:(

unknown_27:($

unknown_28:$

unknown_29:($

unknown_30:$

unknown_31:($

unknown_32:$

unknown_33:($

unknown_34:$

unknown_35:($

unknown_36:$
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallimagesunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36*2
Tin+
)2'*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????$*H
_read_only_resource_inputs*
(&	
 !"#$%&*-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_model_layer_call_and_return_conditional_losses_106344s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:?????????$`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*z
_input_shapesi
g:?????????dd: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????dd
 
_user_specified_nameimages
?
?
Q__inference_batch_normalization_5_layer_call_and_return_conditional_losses_108110

inputs%
readvariableop_resource: '
readvariableop_1_resource: 6
(fusedbatchnormv3_readvariableop_resource: 8
*fusedbatchnormv3_readvariableop_1_resource: 
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype0?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+??????????????????????????? : : : : :*
epsilon%o?:*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+??????????????????????????? ?
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+??????????????????????????? : : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs
?
?
Q__inference_batch_normalization_3_layer_call_and_return_conditional_losses_107926

inputs%
readvariableop_resource: '
readvariableop_1_resource: 6
(fusedbatchnormv3_readvariableop_resource: 8
*fusedbatchnormv3_readvariableop_1_resource: 
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype0?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+??????????????????????????? : : : : :*
epsilon%o?:*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+??????????????????????????? ?
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+??????????????????????????? : : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs
??
?
A__inference_model_layer_call_and_return_conditional_losses_107143

images(
th_conv_107033: 
th_conv_107035: *
batch_normalization_3_107039: *
batch_normalization_3_107041: *
batch_normalization_3_107043: *
batch_normalization_3_107045: (
th_conv_107048:  
th_conv_107050: *
batch_normalization_4_107054: *
batch_normalization_4_107056: *
batch_normalization_4_107058: *
batch_normalization_4_107060: (
th_conv_107063:  
th_conv_107065: *
batch_normalization_5_107069: *
batch_normalization_5_107071: *
batch_normalization_5_107073: *
batch_normalization_5_107075: 1
st_fully_connected_of_4_107079:	?$(,
st_fully_connected_of_4_107081:(1
st_fully_connected_of_3_107084:	?$(,
st_fully_connected_of_3_107086:(1
st_fully_connected_of_2_107089:	?$(,
st_fully_connected_of_2_107091:(1
st_fully_connected_of_1_107094:	?$(,
st_fully_connected_of_1_107096:(1
st_fully_connected_of_0_107099:	?$(,
st_fully_connected_of_0_107101:(!
output_0_107109:($
output_0_107111:$!
output_1_107114:($
output_1_107116:$!
output_2_107119:($
output_2_107121:$!
output_3_107124:($
output_3_107126:$!
output_4_107129:($
output_4_107131:$
identity??01st_fully_connected_of_0/StatefulPartitionedCall?01st_fully_connected_of_1/StatefulPartitionedCall?01st_fully_connected_of_2/StatefulPartitionedCall?01st_fully_connected_of_3/StatefulPartitionedCall?01st_fully_connected_of_4/StatefulPartitionedCall? 4th_conv/StatefulPartitionedCall? 5th_conv/StatefulPartitionedCall? 6th_conv/StatefulPartitionedCall?-batch_normalization_3/StatefulPartitionedCall?-batch_normalization_4/StatefulPartitionedCall?-batch_normalization_5/StatefulPartitionedCall? output_0/StatefulPartitionedCall? output_1/StatefulPartitionedCall? output_2/StatefulPartitionedCall? output_3/StatefulPartitionedCall? output_4/StatefulPartitionedCall?
 4th_conv/StatefulPartitionedCallStatefulPartitionedCallimagesth_conv_107033th_conv_107035*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????dd *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_4th_conv_layer_call_and_return_conditional_losses_106052?
4th_pool/PartitionedCallPartitionedCall)4th_conv/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????22 * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_4th_pool_layer_call_and_return_conditional_losses_105815?
-batch_normalization_3/StatefulPartitionedCallStatefulPartitionedCall!4th_pool/PartitionedCall:output:0batch_normalization_3_107039batch_normalization_3_107041batch_normalization_3_107043batch_normalization_3_107045*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????22 *&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Z
fURS
Q__inference_batch_normalization_3_layer_call_and_return_conditional_losses_105840?
 5th_conv/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_3/StatefulPartitionedCall:output:0th_conv_107048th_conv_107050*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????22 *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_5th_conv_layer_call_and_return_conditional_losses_106079?
5th_pool/PartitionedCallPartitionedCall)5th_conv/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_5th_pool_layer_call_and_return_conditional_losses_105891?
-batch_normalization_4/StatefulPartitionedCallStatefulPartitionedCall!5th_pool/PartitionedCall:output:0batch_normalization_4_107054batch_normalization_4_107056batch_normalization_4_107058batch_normalization_4_107060*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? *&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Z
fURS
Q__inference_batch_normalization_4_layer_call_and_return_conditional_losses_105916?
 6th_conv/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_4/StatefulPartitionedCall:output:0th_conv_107063th_conv_107065*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_6th_conv_layer_call_and_return_conditional_losses_106106?
6th_pool/PartitionedCallPartitionedCall)6th_conv/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_6th_pool_layer_call_and_return_conditional_losses_105967?
-batch_normalization_5/StatefulPartitionedCallStatefulPartitionedCall!6th_pool/PartitionedCall:output:0batch_normalization_5_107069batch_normalization_5_107071batch_normalization_5_107073batch_normalization_5_107075*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? *&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Z
fURS
Q__inference_batch_normalization_5_layer_call_and_return_conditional_losses_105992?
flat/PartitionedCallPartitionedCall6batch_normalization_5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????$* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *I
fDRB
@__inference_flat_layer_call_and_return_conditional_losses_106128?
01st_fully_connected_of_4/StatefulPartitionedCallStatefulPartitionedCallflat/PartitionedCall:output:0st_fully_connected_of_4_107079st_fully_connected_of_4_107081*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????(*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *]
fXRV
T__inference_1st_fully_connected_of_4_layer_call_and_return_conditional_losses_106141?
01st_fully_connected_of_3/StatefulPartitionedCallStatefulPartitionedCallflat/PartitionedCall:output:0st_fully_connected_of_3_107084st_fully_connected_of_3_107086*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????(*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *]
fXRV
T__inference_1st_fully_connected_of_3_layer_call_and_return_conditional_losses_106158?
01st_fully_connected_of_2/StatefulPartitionedCallStatefulPartitionedCallflat/PartitionedCall:output:0st_fully_connected_of_2_107089st_fully_connected_of_2_107091*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????(*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *]
fXRV
T__inference_1st_fully_connected_of_2_layer_call_and_return_conditional_losses_106175?
01st_fully_connected_of_1/StatefulPartitionedCallStatefulPartitionedCallflat/PartitionedCall:output:0st_fully_connected_of_1_107094st_fully_connected_of_1_107096*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????(*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *]
fXRV
T__inference_1st_fully_connected_of_1_layer_call_and_return_conditional_losses_106192?
01st_fully_connected_of_0/StatefulPartitionedCallStatefulPartitionedCallflat/PartitionedCall:output:0st_fully_connected_of_0_107099st_fully_connected_of_0_107101*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????(*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *]
fXRV
T__inference_1st_fully_connected_of_0_layer_call_and_return_conditional_losses_106209?
dropout_4/PartitionedCallPartitionedCall91st_fully_connected_of_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????(* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dropout_4_layer_call_and_return_conditional_losses_106220?
dropout_3/PartitionedCallPartitionedCall91st_fully_connected_of_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????(* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dropout_3_layer_call_and_return_conditional_losses_106227?
dropout_2/PartitionedCallPartitionedCall91st_fully_connected_of_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????(* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dropout_2_layer_call_and_return_conditional_losses_106234?
dropout_1/PartitionedCallPartitionedCall91st_fully_connected_of_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????(* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dropout_1_layer_call_and_return_conditional_losses_106241?
dropout/PartitionedCallPartitionedCall91st_fully_connected_of_0/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????(* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dropout_layer_call_and_return_conditional_losses_106248?
 output_0/StatefulPartitionedCallStatefulPartitionedCall dropout/PartitionedCall:output:0output_0_107109output_0_107111*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????$*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_output_0_layer_call_and_return_conditional_losses_106261?
 output_1/StatefulPartitionedCallStatefulPartitionedCall"dropout_1/PartitionedCall:output:0output_1_107114output_1_107116*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????$*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_output_1_layer_call_and_return_conditional_losses_106278?
 output_2/StatefulPartitionedCallStatefulPartitionedCall"dropout_2/PartitionedCall:output:0output_2_107119output_2_107121*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????$*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_output_2_layer_call_and_return_conditional_losses_106295?
 output_3/StatefulPartitionedCallStatefulPartitionedCall"dropout_3/PartitionedCall:output:0output_3_107124output_3_107126*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????$*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_output_3_layer_call_and_return_conditional_losses_106312?
 output_4/StatefulPartitionedCallStatefulPartitionedCall"dropout_4/PartitionedCall:output:0output_4_107129output_4_107131*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????$*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_output_4_layer_call_and_return_conditional_losses_106329?
tf.convert_to_tensor_1/CastCast)output_0/StatefulPartitionedCall:output:0*

DstT0*

SrcT0*'
_output_shapes
:?????????$?
tf.convert_to_tensor_1/Cast_1Cast)output_1/StatefulPartitionedCall:output:0*

DstT0*

SrcT0*'
_output_shapes
:?????????$?
tf.convert_to_tensor_1/Cast_2Cast)output_2/StatefulPartitionedCall:output:0*

DstT0*

SrcT0*'
_output_shapes
:?????????$?
tf.convert_to_tensor_1/Cast_3Cast)output_3/StatefulPartitionedCall:output:0*

DstT0*

SrcT0*'
_output_shapes
:?????????$?
tf.convert_to_tensor_1/Cast_4Cast)output_4/StatefulPartitionedCall:output:0*

DstT0*

SrcT0*'
_output_shapes
:?????????$?
tf.convert_to_tensor_1/packedPacktf.convert_to_tensor_1/Cast:y:0!tf.convert_to_tensor_1/Cast_1:y:0!tf.convert_to_tensor_1/Cast_2:y:0!tf.convert_to_tensor_1/Cast_3:y:0!tf.convert_to_tensor_1/Cast_4:y:0*
N*
T0*+
_output_shapes
:?????????$|
'tf.compat.v1.transpose_1/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          ?
"tf.compat.v1.transpose_1/transpose	Transpose&tf.convert_to_tensor_1/packed:output:00tf.compat.v1.transpose_1/transpose/perm:output:0*
T0*+
_output_shapes
:?????????$y
IdentityIdentity&tf.compat.v1.transpose_1/transpose:y:0^NoOp*
T0*+
_output_shapes
:?????????$?
NoOpNoOp1^1st_fully_connected_of_0/StatefulPartitionedCall1^1st_fully_connected_of_1/StatefulPartitionedCall1^1st_fully_connected_of_2/StatefulPartitionedCall1^1st_fully_connected_of_3/StatefulPartitionedCall1^1st_fully_connected_of_4/StatefulPartitionedCall!^4th_conv/StatefulPartitionedCall!^5th_conv/StatefulPartitionedCall!^6th_conv/StatefulPartitionedCall.^batch_normalization_3/StatefulPartitionedCall.^batch_normalization_4/StatefulPartitionedCall.^batch_normalization_5/StatefulPartitionedCall!^output_0/StatefulPartitionedCall!^output_1/StatefulPartitionedCall!^output_2/StatefulPartitionedCall!^output_3/StatefulPartitionedCall!^output_4/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*z
_input_shapesi
g:?????????dd: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2d
01st_fully_connected_of_0/StatefulPartitionedCall01st_fully_connected_of_0/StatefulPartitionedCall2d
01st_fully_connected_of_1/StatefulPartitionedCall01st_fully_connected_of_1/StatefulPartitionedCall2d
01st_fully_connected_of_2/StatefulPartitionedCall01st_fully_connected_of_2/StatefulPartitionedCall2d
01st_fully_connected_of_3/StatefulPartitionedCall01st_fully_connected_of_3/StatefulPartitionedCall2d
01st_fully_connected_of_4/StatefulPartitionedCall01st_fully_connected_of_4/StatefulPartitionedCall2D
 4th_conv/StatefulPartitionedCall 4th_conv/StatefulPartitionedCall2D
 5th_conv/StatefulPartitionedCall 5th_conv/StatefulPartitionedCall2D
 6th_conv/StatefulPartitionedCall 6th_conv/StatefulPartitionedCall2^
-batch_normalization_3/StatefulPartitionedCall-batch_normalization_3/StatefulPartitionedCall2^
-batch_normalization_4/StatefulPartitionedCall-batch_normalization_4/StatefulPartitionedCall2^
-batch_normalization_5/StatefulPartitionedCall-batch_normalization_5/StatefulPartitionedCall2D
 output_0/StatefulPartitionedCall output_0/StatefulPartitionedCall2D
 output_1/StatefulPartitionedCall output_1/StatefulPartitionedCall2D
 output_2/StatefulPartitionedCall output_2/StatefulPartitionedCall2D
 output_3/StatefulPartitionedCall output_3/StatefulPartitionedCall2D
 output_4/StatefulPartitionedCall output_4/StatefulPartitionedCall:W S
/
_output_shapes
:?????????dd
 
_user_specified_nameimages
?
A
%__inference_flat_layer_call_fn_108133

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????$* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *I
fDRB
@__inference_flat_layer_call_and_return_conditional_losses_106128a
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:??????????$"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:????????? :W S
/
_output_shapes
:????????? 
 
_user_specified_nameinputs
?
?
9__inference_1st_fully_connected_of_1_layer_call_fn_108168

inputs
unknown:	?$(
	unknown_0:(
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????(*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *]
fXRV
T__inference_1st_fully_connected_of_1_layer_call_and_return_conditional_losses_106192o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????(`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????$: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????$
 
_user_specified_nameinputs
?
?
D__inference_5th_conv_layer_call_and_return_conditional_losses_107964

inputs8
conv2d_readvariableop_resource:  -
biasadd_readvariableop_resource: 
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:  *
dtype0?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????22 *
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????22 X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:?????????22 i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:?????????22 w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????22 : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????22 
 
_user_specified_nameinputs
?
?
&__inference_model_layer_call_fn_107426

inputs!
unknown: 
	unknown_0: 
	unknown_1: 
	unknown_2: 
	unknown_3: 
	unknown_4: #
	unknown_5:  
	unknown_6: 
	unknown_7: 
	unknown_8: 
	unknown_9: 

unknown_10: $

unknown_11:  

unknown_12: 

unknown_13: 

unknown_14: 

unknown_15: 

unknown_16: 

unknown_17:	?$(

unknown_18:(

unknown_19:	?$(

unknown_20:(

unknown_21:	?$(

unknown_22:(

unknown_23:	?$(

unknown_24:(

unknown_25:	?$(

unknown_26:(

unknown_27:($

unknown_28:$

unknown_29:($

unknown_30:$

unknown_31:($

unknown_32:$

unknown_33:($

unknown_34:$

unknown_35:($

unknown_36:$
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36*2
Tin+
)2'*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????$*H
_read_only_resource_inputs*
(&	
 !"#$%&*-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_model_layer_call_and_return_conditional_losses_106344s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:?????????$`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*z
_input_shapesi
g:?????????dd: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????dd
 
_user_specified_nameinputs
?

?
T__inference_1st_fully_connected_of_4_layer_call_and_return_conditional_losses_108239

inputs1
matmul_readvariableop_resource:	?$(-
biasadd_readvariableop_resource:(
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?$(*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????(r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:(*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????(P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:?????????(a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:?????????(w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????$: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????$
 
_user_specified_nameinputs
?
?
Q__inference_batch_normalization_3_layer_call_and_return_conditional_losses_105871

inputs%
readvariableop_resource: '
readvariableop_1_resource: 6
(fusedbatchnormv3_readvariableop_resource: 8
*fusedbatchnormv3_readvariableop_1_resource: 
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype0?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+??????????????????????????? : : : : :*
epsilon%o?:*
exponential_avg_factor%
?#<?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(}
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+??????????????????????????? ?
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+??????????????????????????? : : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs"?L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*?
serving_default?
A
images7
serving_default_images:0?????????ddP
tf.compat.v1.transpose_14
StatefulPartitionedCall:0?????????$tensorflow/serving/predict:??
?
layer-0
layer_with_weights-0
layer-1
layer-2
layer_with_weights-1
layer-3
layer_with_weights-2
layer-4
layer-5
layer_with_weights-3
layer-6
layer_with_weights-4
layer-7
	layer-8

layer_with_weights-5

layer-9
layer-10
layer_with_weights-6
layer-11
layer_with_weights-7
layer-12
layer_with_weights-8
layer-13
layer_with_weights-9
layer-14
layer_with_weights-10
layer-15
layer-16
layer-17
layer-18
layer-19
layer-20
layer_with_weights-11
layer-21
layer_with_weights-12
layer-22
layer_with_weights-13
layer-23
layer_with_weights-14
layer-24
layer_with_weights-15
layer-25
layer-26
layer-27
	variables
trainable_variables
regularization_losses
 	keras_api
!__call__
*"&call_and_return_all_conditional_losses
#_default_save_signature
$	optimizer
%
signatures"
_tf_keras_network
"
_tf_keras_input_layer
?
&	variables
'trainable_variables
(regularization_losses
)	keras_api
*__call__
*+&call_and_return_all_conditional_losses

,kernel
-bias
 ._jit_compiled_convolution_op"
_tf_keras_layer
?
/	variables
0trainable_variables
1regularization_losses
2	keras_api
3__call__
*4&call_and_return_all_conditional_losses"
_tf_keras_layer
?
5	variables
6trainable_variables
7regularization_losses
8	keras_api
9__call__
*:&call_and_return_all_conditional_losses
;axis
	<gamma
=beta
>moving_mean
?moving_variance"
_tf_keras_layer
?
@	variables
Atrainable_variables
Bregularization_losses
C	keras_api
D__call__
*E&call_and_return_all_conditional_losses

Fkernel
Gbias
 H_jit_compiled_convolution_op"
_tf_keras_layer
?
I	variables
Jtrainable_variables
Kregularization_losses
L	keras_api
M__call__
*N&call_and_return_all_conditional_losses"
_tf_keras_layer
?
O	variables
Ptrainable_variables
Qregularization_losses
R	keras_api
S__call__
*T&call_and_return_all_conditional_losses
Uaxis
	Vgamma
Wbeta
Xmoving_mean
Ymoving_variance"
_tf_keras_layer
?
Z	variables
[trainable_variables
\regularization_losses
]	keras_api
^__call__
*_&call_and_return_all_conditional_losses

`kernel
abias
 b_jit_compiled_convolution_op"
_tf_keras_layer
?
c	variables
dtrainable_variables
eregularization_losses
f	keras_api
g__call__
*h&call_and_return_all_conditional_losses"
_tf_keras_layer
?
i	variables
jtrainable_variables
kregularization_losses
l	keras_api
m__call__
*n&call_and_return_all_conditional_losses
oaxis
	pgamma
qbeta
rmoving_mean
smoving_variance"
_tf_keras_layer
?
t	variables
utrainable_variables
vregularization_losses
w	keras_api
x__call__
*y&call_and_return_all_conditional_losses"
_tf_keras_layer
?
z	variables
{trainable_variables
|regularization_losses
}	keras_api
~__call__
*&call_and_return_all_conditional_losses
?kernel
	?bias"
_tf_keras_layer
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses
?kernel
	?bias"
_tf_keras_layer
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses
?kernel
	?bias"
_tf_keras_layer
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses
?kernel
	?bias"
_tf_keras_layer
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses
?kernel
	?bias"
_tf_keras_layer
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses
?_random_generator"
_tf_keras_layer
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses
?_random_generator"
_tf_keras_layer
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses
?_random_generator"
_tf_keras_layer
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses
?_random_generator"
_tf_keras_layer
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses
?_random_generator"
_tf_keras_layer
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses
?kernel
	?bias"
_tf_keras_layer
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses
?kernel
	?bias"
_tf_keras_layer
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses
?kernel
	?bias"
_tf_keras_layer
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses
?kernel
	?bias"
_tf_keras_layer
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses
?kernel
	?bias"
_tf_keras_layer
)
?	keras_api"
_tf_keras_layer
)
?	keras_api"
_tf_keras_layer
?
,0
-1
<2
=3
>4
?5
F6
G7
V8
W9
X10
Y11
`12
a13
p14
q15
r16
s17
?18
?19
?20
?21
?22
?23
?24
?25
?26
?27
?28
?29
?30
?31
?32
?33
?34
?35
?36
?37"
trackable_list_wrapper
?
,0
-1
<2
=3
F4
G5
V6
W7
`8
a9
p10
q11
?12
?13
?14
?15
?16
?17
?18
?19
?20
?21
?22
?23
?24
?25
?26
?27
?28
?29
?30
?31"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
	variables
trainable_variables
regularization_losses
!__call__
#_default_save_signature
*"&call_and_return_all_conditional_losses
&""call_and_return_conditional_losses"
_generic_user_object
?
?trace_0
?trace_1
?trace_2
?trace_32?
&__inference_model_layer_call_fn_106423
&__inference_model_layer_call_fn_107426
&__inference_model_layer_call_fn_107507
&__inference_model_layer_call_fn_107030?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 z?trace_0z?trace_1z?trace_2z?trace_3
?
?trace_0
?trace_1
?trace_2
?trace_32?
A__inference_model_layer_call_and_return_conditional_losses_107662
A__inference_model_layer_call_and_return_conditional_losses_107852
A__inference_model_layer_call_and_return_conditional_losses_107143
A__inference_model_layer_call_and_return_conditional_losses_107256?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 z?trace_0z?trace_1z?trace_2z?trace_3
?B?
!__inference__wrapped_model_105806images"?
???
FullArgSpec
args? 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?
	?iter
?beta_1
?beta_2

?decay
?learning_rate,m?-m?<m?=m?Fm?Gm?Vm?Wm?`m?am?pm?qm?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?,v?-v?<v?=v?Fv?Gv?Vv?Wv?`v?av?pv?qv?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?"
	optimizer
-
?serving_default"
signature_map
.
,0
-1"
trackable_list_wrapper
.
,0
-1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
&	variables
'trainable_variables
(regularization_losses
*__call__
*+&call_and_return_all_conditional_losses
&+"call_and_return_conditional_losses"
_generic_user_object
?
?trace_02?
)__inference_4th_conv_layer_call_fn_107861?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0
?
?trace_02?
D__inference_4th_conv_layer_call_and_return_conditional_losses_107872?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0
):' 24th_conv/kernel
: 24th_conv/bias
?2??
???
FullArgSpec'
args?
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
/	variables
0trainable_variables
1regularization_losses
3__call__
*4&call_and_return_all_conditional_losses
&4"call_and_return_conditional_losses"
_generic_user_object
?
?trace_02?
)__inference_4th_pool_layer_call_fn_107877?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0
?
?trace_02?
D__inference_4th_pool_layer_call_and_return_conditional_losses_107882?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0
<
<0
=1
>2
?3"
trackable_list_wrapper
.
<0
=1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
5	variables
6trainable_variables
7regularization_losses
9__call__
*:&call_and_return_all_conditional_losses
&:"call_and_return_conditional_losses"
_generic_user_object
?
?trace_0
?trace_12?
6__inference_batch_normalization_3_layer_call_fn_107895
6__inference_batch_normalization_3_layer_call_fn_107908?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 z?trace_0z?trace_1
?
?trace_0
?trace_12?
Q__inference_batch_normalization_3_layer_call_and_return_conditional_losses_107926
Q__inference_batch_normalization_3_layer_call_and_return_conditional_losses_107944?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 z?trace_0z?trace_1
 "
trackable_list_wrapper
):' 2batch_normalization_3/gamma
(:& 2batch_normalization_3/beta
1:/  (2!batch_normalization_3/moving_mean
5:3  (2%batch_normalization_3/moving_variance
.
F0
G1"
trackable_list_wrapper
.
F0
G1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
@	variables
Atrainable_variables
Bregularization_losses
D__call__
*E&call_and_return_all_conditional_losses
&E"call_and_return_conditional_losses"
_generic_user_object
?
?trace_02?
)__inference_5th_conv_layer_call_fn_107953?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0
?
?trace_02?
D__inference_5th_conv_layer_call_and_return_conditional_losses_107964?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0
):'  25th_conv/kernel
: 25th_conv/bias
?2??
???
FullArgSpec'
args?
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
I	variables
Jtrainable_variables
Kregularization_losses
M__call__
*N&call_and_return_all_conditional_losses
&N"call_and_return_conditional_losses"
_generic_user_object
?
?trace_02?
)__inference_5th_pool_layer_call_fn_107969?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0
?
?trace_02?
D__inference_5th_pool_layer_call_and_return_conditional_losses_107974?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0
<
V0
W1
X2
Y3"
trackable_list_wrapper
.
V0
W1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
O	variables
Ptrainable_variables
Qregularization_losses
S__call__
*T&call_and_return_all_conditional_losses
&T"call_and_return_conditional_losses"
_generic_user_object
?
?trace_0
?trace_12?
6__inference_batch_normalization_4_layer_call_fn_107987
6__inference_batch_normalization_4_layer_call_fn_108000?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 z?trace_0z?trace_1
?
?trace_0
?trace_12?
Q__inference_batch_normalization_4_layer_call_and_return_conditional_losses_108018
Q__inference_batch_normalization_4_layer_call_and_return_conditional_losses_108036?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 z?trace_0z?trace_1
 "
trackable_list_wrapper
):' 2batch_normalization_4/gamma
(:& 2batch_normalization_4/beta
1:/  (2!batch_normalization_4/moving_mean
5:3  (2%batch_normalization_4/moving_variance
.
`0
a1"
trackable_list_wrapper
.
`0
a1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
Z	variables
[trainable_variables
\regularization_losses
^__call__
*_&call_and_return_all_conditional_losses
&_"call_and_return_conditional_losses"
_generic_user_object
?
?trace_02?
)__inference_6th_conv_layer_call_fn_108045?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0
?
?trace_02?
D__inference_6th_conv_layer_call_and_return_conditional_losses_108056?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0
):'  26th_conv/kernel
: 26th_conv/bias
?2??
???
FullArgSpec'
args?
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
c	variables
dtrainable_variables
eregularization_losses
g__call__
*h&call_and_return_all_conditional_losses
&h"call_and_return_conditional_losses"
_generic_user_object
?
?trace_02?
)__inference_6th_pool_layer_call_fn_108061?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0
?
?trace_02?
D__inference_6th_pool_layer_call_and_return_conditional_losses_108066?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0
<
p0
q1
r2
s3"
trackable_list_wrapper
.
p0
q1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
i	variables
jtrainable_variables
kregularization_losses
m__call__
*n&call_and_return_all_conditional_losses
&n"call_and_return_conditional_losses"
_generic_user_object
?
?trace_0
?trace_12?
6__inference_batch_normalization_5_layer_call_fn_108079
6__inference_batch_normalization_5_layer_call_fn_108092?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 z?trace_0z?trace_1
?
?trace_0
?trace_12?
Q__inference_batch_normalization_5_layer_call_and_return_conditional_losses_108110
Q__inference_batch_normalization_5_layer_call_and_return_conditional_losses_108128?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 z?trace_0z?trace_1
 "
trackable_list_wrapper
):' 2batch_normalization_5/gamma
(:& 2batch_normalization_5/beta
1:/  (2!batch_normalization_5/moving_mean
5:3  (2%batch_normalization_5/moving_variance
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
t	variables
utrainable_variables
vregularization_losses
x__call__
*y&call_and_return_all_conditional_losses
&y"call_and_return_conditional_losses"
_generic_user_object
?
?trace_02?
%__inference_flat_layer_call_fn_108133?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0
?
?trace_02?
@__inference_flat_layer_call_and_return_conditional_losses_108139?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0
0
?0
?1"
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
z	variables
{trainable_variables
|regularization_losses
~__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
?
?trace_02?
9__inference_1st_fully_connected_of_0_layer_call_fn_108148?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0
?
?trace_02?
T__inference_1st_fully_connected_of_0_layer_call_and_return_conditional_losses_108159?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0
2:0	?$(21st_fully_connected_of_0/kernel
+:)(21st_fully_connected_of_0/bias
0
?0
?1"
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?
?trace_02?
9__inference_1st_fully_connected_of_1_layer_call_fn_108168?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0
?
?trace_02?
T__inference_1st_fully_connected_of_1_layer_call_and_return_conditional_losses_108179?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0
2:0	?$(21st_fully_connected_of_1/kernel
+:)(21st_fully_connected_of_1/bias
0
?0
?1"
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?
?trace_02?
9__inference_1st_fully_connected_of_2_layer_call_fn_108188?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0
?
?trace_02?
T__inference_1st_fully_connected_of_2_layer_call_and_return_conditional_losses_108199?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0
2:0	?$(21st_fully_connected_of_2/kernel
+:)(21st_fully_connected_of_2/bias
0
?0
?1"
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?
?trace_02?
9__inference_1st_fully_connected_of_3_layer_call_fn_108208?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0
?
?trace_02?
T__inference_1st_fully_connected_of_3_layer_call_and_return_conditional_losses_108219?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0
2:0	?$(21st_fully_connected_of_3/kernel
+:)(21st_fully_connected_of_3/bias
0
?0
?1"
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?
?trace_02?
9__inference_1st_fully_connected_of_4_layer_call_fn_108228?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0
?
?trace_02?
T__inference_1st_fully_connected_of_4_layer_call_and_return_conditional_losses_108239?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0
2:0	?$(21st_fully_connected_of_4/kernel
+:)(21st_fully_connected_of_4/bias
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?
?trace_0
?trace_12?
(__inference_dropout_layer_call_fn_108244
(__inference_dropout_layer_call_fn_108249?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 z?trace_0z?trace_1
?
?trace_0
?trace_12?
C__inference_dropout_layer_call_and_return_conditional_losses_108254
C__inference_dropout_layer_call_and_return_conditional_losses_108266?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 z?trace_0z?trace_1
"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?
?trace_0
?trace_12?
*__inference_dropout_1_layer_call_fn_108271
*__inference_dropout_1_layer_call_fn_108276?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 z?trace_0z?trace_1
?
?trace_0
?trace_12?
E__inference_dropout_1_layer_call_and_return_conditional_losses_108281
E__inference_dropout_1_layer_call_and_return_conditional_losses_108293?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 z?trace_0z?trace_1
"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?
?trace_0
?trace_12?
*__inference_dropout_2_layer_call_fn_108298
*__inference_dropout_2_layer_call_fn_108303?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 z?trace_0z?trace_1
?
?trace_0
?trace_12?
E__inference_dropout_2_layer_call_and_return_conditional_losses_108308
E__inference_dropout_2_layer_call_and_return_conditional_losses_108320?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 z?trace_0z?trace_1
"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?
?trace_0
?trace_12?
*__inference_dropout_3_layer_call_fn_108325
*__inference_dropout_3_layer_call_fn_108330?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 z?trace_0z?trace_1
?
?trace_0
?trace_12?
E__inference_dropout_3_layer_call_and_return_conditional_losses_108335
E__inference_dropout_3_layer_call_and_return_conditional_losses_108347?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 z?trace_0z?trace_1
"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?
?trace_0
?trace_12?
*__inference_dropout_4_layer_call_fn_108352
*__inference_dropout_4_layer_call_fn_108357?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 z?trace_0z?trace_1
?
?trace_0
?trace_12?
E__inference_dropout_4_layer_call_and_return_conditional_losses_108362
E__inference_dropout_4_layer_call_and_return_conditional_losses_108374?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 z?trace_0z?trace_1
"
_generic_user_object
0
?0
?1"
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?
?trace_02?
)__inference_output_0_layer_call_fn_108383?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0
?
?trace_02?
D__inference_output_0_layer_call_and_return_conditional_losses_108394?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0
!:($2output_0/kernel
:$2output_0/bias
0
?0
?1"
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?
?trace_02?
)__inference_output_1_layer_call_fn_108403?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0
?
?trace_02?
D__inference_output_1_layer_call_and_return_conditional_losses_108414?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0
!:($2output_1/kernel
:$2output_1/bias
0
?0
?1"
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?
?trace_02?
)__inference_output_2_layer_call_fn_108423?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0
?
?trace_02?
D__inference_output_2_layer_call_and_return_conditional_losses_108434?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0
!:($2output_2/kernel
:$2output_2/bias
0
?0
?1"
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?
?trace_02?
)__inference_output_3_layer_call_fn_108443?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0
?
?trace_02?
D__inference_output_3_layer_call_and_return_conditional_losses_108454?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0
!:($2output_3/kernel
:$2output_3/bias
0
?0
?1"
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?
?trace_02?
)__inference_output_4_layer_call_fn_108463?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0
?
?trace_02?
D__inference_output_4_layer_call_and_return_conditional_losses_108474?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0
!:($2output_4/kernel
:$2output_4/bias
"
_generic_user_object
"
_generic_user_object
J
>0
?1
X2
Y3
r4
s5"
trackable_list_wrapper
?
0
1
2
3
4
5
6
7
	8

9
10
11
12
13
14
15
16
17
18
19
20
21
22
23
24
25
26
27"
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
?B?
&__inference_model_layer_call_fn_106423images"?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
&__inference_model_layer_call_fn_107426inputs"?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
&__inference_model_layer_call_fn_107507inputs"?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
&__inference_model_layer_call_fn_107030images"?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
A__inference_model_layer_call_and_return_conditional_losses_107662inputs"?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
A__inference_model_layer_call_and_return_conditional_losses_107852inputs"?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
A__inference_model_layer_call_and_return_conditional_losses_107143images"?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
A__inference_model_layer_call_and_return_conditional_losses_107256images"?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
?B?
$__inference_signature_wrapper_107345images"?
???
FullArgSpec
args? 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
?B?
)__inference_4th_conv_layer_call_fn_107861inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
D__inference_4th_conv_layer_call_and_return_conditional_losses_107872inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
?B?
)__inference_4th_pool_layer_call_fn_107877inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
D__inference_4th_pool_layer_call_and_return_conditional_losses_107882inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
.
>0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
?B?
6__inference_batch_normalization_3_layer_call_fn_107895inputs"?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
6__inference_batch_normalization_3_layer_call_fn_107908inputs"?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
Q__inference_batch_normalization_3_layer_call_and_return_conditional_losses_107926inputs"?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
Q__inference_batch_normalization_3_layer_call_and_return_conditional_losses_107944inputs"?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
?B?
)__inference_5th_conv_layer_call_fn_107953inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
D__inference_5th_conv_layer_call_and_return_conditional_losses_107964inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
?B?
)__inference_5th_pool_layer_call_fn_107969inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
D__inference_5th_pool_layer_call_and_return_conditional_losses_107974inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
.
X0
Y1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
?B?
6__inference_batch_normalization_4_layer_call_fn_107987inputs"?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
6__inference_batch_normalization_4_layer_call_fn_108000inputs"?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
Q__inference_batch_normalization_4_layer_call_and_return_conditional_losses_108018inputs"?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
Q__inference_batch_normalization_4_layer_call_and_return_conditional_losses_108036inputs"?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
?B?
)__inference_6th_conv_layer_call_fn_108045inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
D__inference_6th_conv_layer_call_and_return_conditional_losses_108056inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
?B?
)__inference_6th_pool_layer_call_fn_108061inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
D__inference_6th_pool_layer_call_and_return_conditional_losses_108066inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
.
r0
s1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
?B?
6__inference_batch_normalization_5_layer_call_fn_108079inputs"?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
6__inference_batch_normalization_5_layer_call_fn_108092inputs"?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
Q__inference_batch_normalization_5_layer_call_and_return_conditional_losses_108110inputs"?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
Q__inference_batch_normalization_5_layer_call_and_return_conditional_losses_108128inputs"?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
?B?
%__inference_flat_layer_call_fn_108133inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
@__inference_flat_layer_call_and_return_conditional_losses_108139inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
?B?
9__inference_1st_fully_connected_of_0_layer_call_fn_108148inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
T__inference_1st_fully_connected_of_0_layer_call_and_return_conditional_losses_108159inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
?B?
9__inference_1st_fully_connected_of_1_layer_call_fn_108168inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
T__inference_1st_fully_connected_of_1_layer_call_and_return_conditional_losses_108179inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
?B?
9__inference_1st_fully_connected_of_2_layer_call_fn_108188inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
T__inference_1st_fully_connected_of_2_layer_call_and_return_conditional_losses_108199inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
?B?
9__inference_1st_fully_connected_of_3_layer_call_fn_108208inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
T__inference_1st_fully_connected_of_3_layer_call_and_return_conditional_losses_108219inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
?B?
9__inference_1st_fully_connected_of_4_layer_call_fn_108228inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
T__inference_1st_fully_connected_of_4_layer_call_and_return_conditional_losses_108239inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
?B?
(__inference_dropout_layer_call_fn_108244inputs"?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
(__inference_dropout_layer_call_fn_108249inputs"?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
C__inference_dropout_layer_call_and_return_conditional_losses_108254inputs"?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
C__inference_dropout_layer_call_and_return_conditional_losses_108266inputs"?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
?B?
*__inference_dropout_1_layer_call_fn_108271inputs"?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
*__inference_dropout_1_layer_call_fn_108276inputs"?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
E__inference_dropout_1_layer_call_and_return_conditional_losses_108281inputs"?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
E__inference_dropout_1_layer_call_and_return_conditional_losses_108293inputs"?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
?B?
*__inference_dropout_2_layer_call_fn_108298inputs"?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
*__inference_dropout_2_layer_call_fn_108303inputs"?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
E__inference_dropout_2_layer_call_and_return_conditional_losses_108308inputs"?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
E__inference_dropout_2_layer_call_and_return_conditional_losses_108320inputs"?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
?B?
*__inference_dropout_3_layer_call_fn_108325inputs"?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
*__inference_dropout_3_layer_call_fn_108330inputs"?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
E__inference_dropout_3_layer_call_and_return_conditional_losses_108335inputs"?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
E__inference_dropout_3_layer_call_and_return_conditional_losses_108347inputs"?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
?B?
*__inference_dropout_4_layer_call_fn_108352inputs"?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
*__inference_dropout_4_layer_call_fn_108357inputs"?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
E__inference_dropout_4_layer_call_and_return_conditional_losses_108362inputs"?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
E__inference_dropout_4_layer_call_and_return_conditional_losses_108374inputs"?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
?B?
)__inference_output_0_layer_call_fn_108383inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
D__inference_output_0_layer_call_and_return_conditional_losses_108394inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
?B?
)__inference_output_1_layer_call_fn_108403inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
D__inference_output_1_layer_call_and_return_conditional_losses_108414inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
?B?
)__inference_output_2_layer_call_fn_108423inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
D__inference_output_2_layer_call_and_return_conditional_losses_108434inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
?B?
)__inference_output_3_layer_call_fn_108443inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
D__inference_output_3_layer_call_and_return_conditional_losses_108454inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
?B?
)__inference_output_4_layer_call_fn_108463inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
D__inference_output_4_layer_call_and_return_conditional_losses_108474inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
R
?	variables
?	keras_api

?total

?count"
_tf_keras_metric
c
?	variables
?	keras_api

?total

?count
?
_fn_kwargs"
_tf_keras_metric
0
?0
?1"
trackable_list_wrapper
.
?	variables"
_generic_user_object
:  (2total
:  (2count
0
?0
?1"
trackable_list_wrapper
.
?	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
.:, 2Adam/4th_conv/kernel/m
 : 2Adam/4th_conv/bias/m
.:, 2"Adam/batch_normalization_3/gamma/m
-:+ 2!Adam/batch_normalization_3/beta/m
.:,  2Adam/5th_conv/kernel/m
 : 2Adam/5th_conv/bias/m
.:, 2"Adam/batch_normalization_4/gamma/m
-:+ 2!Adam/batch_normalization_4/beta/m
.:,  2Adam/6th_conv/kernel/m
 : 2Adam/6th_conv/bias/m
.:, 2"Adam/batch_normalization_5/gamma/m
-:+ 2!Adam/batch_normalization_5/beta/m
7:5	?$(2&Adam/1st_fully_connected_of_0/kernel/m
0:.(2$Adam/1st_fully_connected_of_0/bias/m
7:5	?$(2&Adam/1st_fully_connected_of_1/kernel/m
0:.(2$Adam/1st_fully_connected_of_1/bias/m
7:5	?$(2&Adam/1st_fully_connected_of_2/kernel/m
0:.(2$Adam/1st_fully_connected_of_2/bias/m
7:5	?$(2&Adam/1st_fully_connected_of_3/kernel/m
0:.(2$Adam/1st_fully_connected_of_3/bias/m
7:5	?$(2&Adam/1st_fully_connected_of_4/kernel/m
0:.(2$Adam/1st_fully_connected_of_4/bias/m
&:$($2Adam/output_0/kernel/m
 :$2Adam/output_0/bias/m
&:$($2Adam/output_1/kernel/m
 :$2Adam/output_1/bias/m
&:$($2Adam/output_2/kernel/m
 :$2Adam/output_2/bias/m
&:$($2Adam/output_3/kernel/m
 :$2Adam/output_3/bias/m
&:$($2Adam/output_4/kernel/m
 :$2Adam/output_4/bias/m
.:, 2Adam/4th_conv/kernel/v
 : 2Adam/4th_conv/bias/v
.:, 2"Adam/batch_normalization_3/gamma/v
-:+ 2!Adam/batch_normalization_3/beta/v
.:,  2Adam/5th_conv/kernel/v
 : 2Adam/5th_conv/bias/v
.:, 2"Adam/batch_normalization_4/gamma/v
-:+ 2!Adam/batch_normalization_4/beta/v
.:,  2Adam/6th_conv/kernel/v
 : 2Adam/6th_conv/bias/v
.:, 2"Adam/batch_normalization_5/gamma/v
-:+ 2!Adam/batch_normalization_5/beta/v
7:5	?$(2&Adam/1st_fully_connected_of_0/kernel/v
0:.(2$Adam/1st_fully_connected_of_0/bias/v
7:5	?$(2&Adam/1st_fully_connected_of_1/kernel/v
0:.(2$Adam/1st_fully_connected_of_1/bias/v
7:5	?$(2&Adam/1st_fully_connected_of_2/kernel/v
0:.(2$Adam/1st_fully_connected_of_2/bias/v
7:5	?$(2&Adam/1st_fully_connected_of_3/kernel/v
0:.(2$Adam/1st_fully_connected_of_3/bias/v
7:5	?$(2&Adam/1st_fully_connected_of_4/kernel/v
0:.(2$Adam/1st_fully_connected_of_4/bias/v
&:$($2Adam/output_0/kernel/v
 :$2Adam/output_0/bias/v
&:$($2Adam/output_1/kernel/v
 :$2Adam/output_1/bias/v
&:$($2Adam/output_2/kernel/v
 :$2Adam/output_2/bias/v
&:$($2Adam/output_3/kernel/v
 :$2Adam/output_3/bias/v
&:$($2Adam/output_4/kernel/v
 :$2Adam/output_4/bias/v?
T__inference_1st_fully_connected_of_0_layer_call_and_return_conditional_losses_108159_??0?-
&?#
!?
inputs??????????$
? "%?"
?
0?????????(
? ?
9__inference_1st_fully_connected_of_0_layer_call_fn_108148R??0?-
&?#
!?
inputs??????????$
? "??????????(?
T__inference_1st_fully_connected_of_1_layer_call_and_return_conditional_losses_108179_??0?-
&?#
!?
inputs??????????$
? "%?"
?
0?????????(
? ?
9__inference_1st_fully_connected_of_1_layer_call_fn_108168R??0?-
&?#
!?
inputs??????????$
? "??????????(?
T__inference_1st_fully_connected_of_2_layer_call_and_return_conditional_losses_108199_??0?-
&?#
!?
inputs??????????$
? "%?"
?
0?????????(
? ?
9__inference_1st_fully_connected_of_2_layer_call_fn_108188R??0?-
&?#
!?
inputs??????????$
? "??????????(?
T__inference_1st_fully_connected_of_3_layer_call_and_return_conditional_losses_108219_??0?-
&?#
!?
inputs??????????$
? "%?"
?
0?????????(
? ?
9__inference_1st_fully_connected_of_3_layer_call_fn_108208R??0?-
&?#
!?
inputs??????????$
? "??????????(?
T__inference_1st_fully_connected_of_4_layer_call_and_return_conditional_losses_108239_??0?-
&?#
!?
inputs??????????$
? "%?"
?
0?????????(
? ?
9__inference_1st_fully_connected_of_4_layer_call_fn_108228R??0?-
&?#
!?
inputs??????????$
? "??????????(?
D__inference_4th_conv_layer_call_and_return_conditional_losses_107872l,-7?4
-?*
(?%
inputs?????????dd
? "-?*
#? 
0?????????dd 
? ?
)__inference_4th_conv_layer_call_fn_107861_,-7?4
-?*
(?%
inputs?????????dd
? " ??????????dd ?
D__inference_4th_pool_layer_call_and_return_conditional_losses_107882?R?O
H?E
C?@
inputs4????????????????????????????????????
? "H?E
>?;
04????????????????????????????????????
? ?
)__inference_4th_pool_layer_call_fn_107877?R?O
H?E
C?@
inputs4????????????????????????????????????
? ";?84?????????????????????????????????????
D__inference_5th_conv_layer_call_and_return_conditional_losses_107964lFG7?4
-?*
(?%
inputs?????????22 
? "-?*
#? 
0?????????22 
? ?
)__inference_5th_conv_layer_call_fn_107953_FG7?4
-?*
(?%
inputs?????????22 
? " ??????????22 ?
D__inference_5th_pool_layer_call_and_return_conditional_losses_107974?R?O
H?E
C?@
inputs4????????????????????????????????????
? "H?E
>?;
04????????????????????????????????????
? ?
)__inference_5th_pool_layer_call_fn_107969?R?O
H?E
C?@
inputs4????????????????????????????????????
? ";?84?????????????????????????????????????
D__inference_6th_conv_layer_call_and_return_conditional_losses_108056l`a7?4
-?*
(?%
inputs????????? 
? "-?*
#? 
0????????? 
? ?
)__inference_6th_conv_layer_call_fn_108045_`a7?4
-?*
(?%
inputs????????? 
? " ?????????? ?
D__inference_6th_pool_layer_call_and_return_conditional_losses_108066?R?O
H?E
C?@
inputs4????????????????????????????????????
? "H?E
>?;
04????????????????????????????????????
? ?
)__inference_6th_pool_layer_call_fn_108061?R?O
H?E
C?@
inputs4????????????????????????????????????
? ";?84?????????????????????????????????????
!__inference__wrapped_model_105806?:,-<=>?FGVWXY`apqrs????????????????????7?4
-?*
(?%
images?????????dd
? "W?T
R
tf.compat.v1.transpose_16?3
tf.compat.v1.transpose_1?????????$?
Q__inference_batch_normalization_3_layer_call_and_return_conditional_losses_107926?<=>?M?J
C?@
:?7
inputs+??????????????????????????? 
p 
? "??<
5?2
0+??????????????????????????? 
? ?
Q__inference_batch_normalization_3_layer_call_and_return_conditional_losses_107944?<=>?M?J
C?@
:?7
inputs+??????????????????????????? 
p
? "??<
5?2
0+??????????????????????????? 
? ?
6__inference_batch_normalization_3_layer_call_fn_107895?<=>?M?J
C?@
:?7
inputs+??????????????????????????? 
p 
? "2?/+??????????????????????????? ?
6__inference_batch_normalization_3_layer_call_fn_107908?<=>?M?J
C?@
:?7
inputs+??????????????????????????? 
p
? "2?/+??????????????????????????? ?
Q__inference_batch_normalization_4_layer_call_and_return_conditional_losses_108018?VWXYM?J
C?@
:?7
inputs+??????????????????????????? 
p 
? "??<
5?2
0+??????????????????????????? 
? ?
Q__inference_batch_normalization_4_layer_call_and_return_conditional_losses_108036?VWXYM?J
C?@
:?7
inputs+??????????????????????????? 
p
? "??<
5?2
0+??????????????????????????? 
? ?
6__inference_batch_normalization_4_layer_call_fn_107987?VWXYM?J
C?@
:?7
inputs+??????????????????????????? 
p 
? "2?/+??????????????????????????? ?
6__inference_batch_normalization_4_layer_call_fn_108000?VWXYM?J
C?@
:?7
inputs+??????????????????????????? 
p
? "2?/+??????????????????????????? ?
Q__inference_batch_normalization_5_layer_call_and_return_conditional_losses_108110?pqrsM?J
C?@
:?7
inputs+??????????????????????????? 
p 
? "??<
5?2
0+??????????????????????????? 
? ?
Q__inference_batch_normalization_5_layer_call_and_return_conditional_losses_108128?pqrsM?J
C?@
:?7
inputs+??????????????????????????? 
p
? "??<
5?2
0+??????????????????????????? 
? ?
6__inference_batch_normalization_5_layer_call_fn_108079?pqrsM?J
C?@
:?7
inputs+??????????????????????????? 
p 
? "2?/+??????????????????????????? ?
6__inference_batch_normalization_5_layer_call_fn_108092?pqrsM?J
C?@
:?7
inputs+??????????????????????????? 
p
? "2?/+??????????????????????????? ?
E__inference_dropout_1_layer_call_and_return_conditional_losses_108281\3?0
)?&
 ?
inputs?????????(
p 
? "%?"
?
0?????????(
? ?
E__inference_dropout_1_layer_call_and_return_conditional_losses_108293\3?0
)?&
 ?
inputs?????????(
p
? "%?"
?
0?????????(
? }
*__inference_dropout_1_layer_call_fn_108271O3?0
)?&
 ?
inputs?????????(
p 
? "??????????(}
*__inference_dropout_1_layer_call_fn_108276O3?0
)?&
 ?
inputs?????????(
p
? "??????????(?
E__inference_dropout_2_layer_call_and_return_conditional_losses_108308\3?0
)?&
 ?
inputs?????????(
p 
? "%?"
?
0?????????(
? ?
E__inference_dropout_2_layer_call_and_return_conditional_losses_108320\3?0
)?&
 ?
inputs?????????(
p
? "%?"
?
0?????????(
? }
*__inference_dropout_2_layer_call_fn_108298O3?0
)?&
 ?
inputs?????????(
p 
? "??????????(}
*__inference_dropout_2_layer_call_fn_108303O3?0
)?&
 ?
inputs?????????(
p
? "??????????(?
E__inference_dropout_3_layer_call_and_return_conditional_losses_108335\3?0
)?&
 ?
inputs?????????(
p 
? "%?"
?
0?????????(
? ?
E__inference_dropout_3_layer_call_and_return_conditional_losses_108347\3?0
)?&
 ?
inputs?????????(
p
? "%?"
?
0?????????(
? }
*__inference_dropout_3_layer_call_fn_108325O3?0
)?&
 ?
inputs?????????(
p 
? "??????????(}
*__inference_dropout_3_layer_call_fn_108330O3?0
)?&
 ?
inputs?????????(
p
? "??????????(?
E__inference_dropout_4_layer_call_and_return_conditional_losses_108362\3?0
)?&
 ?
inputs?????????(
p 
? "%?"
?
0?????????(
? ?
E__inference_dropout_4_layer_call_and_return_conditional_losses_108374\3?0
)?&
 ?
inputs?????????(
p
? "%?"
?
0?????????(
? }
*__inference_dropout_4_layer_call_fn_108352O3?0
)?&
 ?
inputs?????????(
p 
? "??????????(}
*__inference_dropout_4_layer_call_fn_108357O3?0
)?&
 ?
inputs?????????(
p
? "??????????(?
C__inference_dropout_layer_call_and_return_conditional_losses_108254\3?0
)?&
 ?
inputs?????????(
p 
? "%?"
?
0?????????(
? ?
C__inference_dropout_layer_call_and_return_conditional_losses_108266\3?0
)?&
 ?
inputs?????????(
p
? "%?"
?
0?????????(
? {
(__inference_dropout_layer_call_fn_108244O3?0
)?&
 ?
inputs?????????(
p 
? "??????????({
(__inference_dropout_layer_call_fn_108249O3?0
)?&
 ?
inputs?????????(
p
? "??????????(?
@__inference_flat_layer_call_and_return_conditional_losses_108139a7?4
-?*
(?%
inputs????????? 
? "&?#
?
0??????????$
? }
%__inference_flat_layer_call_fn_108133T7?4
-?*
(?%
inputs????????? 
? "???????????$?
A__inference_model_layer_call_and_return_conditional_losses_107143?:,-<=>?FGVWXY`apqrs??????????????????????<
5?2
(?%
images?????????dd
p 

 
? ")?&
?
0?????????$
? ?
A__inference_model_layer_call_and_return_conditional_losses_107256?:,-<=>?FGVWXY`apqrs??????????????????????<
5?2
(?%
images?????????dd
p

 
? ")?&
?
0?????????$
? ?
A__inference_model_layer_call_and_return_conditional_losses_107662?:,-<=>?FGVWXY`apqrs??????????????????????<
5?2
(?%
inputs?????????dd
p 

 
? ")?&
?
0?????????$
? ?
A__inference_model_layer_call_and_return_conditional_losses_107852?:,-<=>?FGVWXY`apqrs??????????????????????<
5?2
(?%
inputs?????????dd
p

 
? ")?&
?
0?????????$
? ?
&__inference_model_layer_call_fn_106423?:,-<=>?FGVWXY`apqrs??????????????????????<
5?2
(?%
images?????????dd
p 

 
? "??????????$?
&__inference_model_layer_call_fn_107030?:,-<=>?FGVWXY`apqrs??????????????????????<
5?2
(?%
images?????????dd
p

 
? "??????????$?
&__inference_model_layer_call_fn_107426?:,-<=>?FGVWXY`apqrs??????????????????????<
5?2
(?%
inputs?????????dd
p 

 
? "??????????$?
&__inference_model_layer_call_fn_107507?:,-<=>?FGVWXY`apqrs??????????????????????<
5?2
(?%
inputs?????????dd
p

 
? "??????????$?
D__inference_output_0_layer_call_and_return_conditional_losses_108394^??/?,
%?"
 ?
inputs?????????(
? "%?"
?
0?????????$
? ~
)__inference_output_0_layer_call_fn_108383Q??/?,
%?"
 ?
inputs?????????(
? "??????????$?
D__inference_output_1_layer_call_and_return_conditional_losses_108414^??/?,
%?"
 ?
inputs?????????(
? "%?"
?
0?????????$
? ~
)__inference_output_1_layer_call_fn_108403Q??/?,
%?"
 ?
inputs?????????(
? "??????????$?
D__inference_output_2_layer_call_and_return_conditional_losses_108434^??/?,
%?"
 ?
inputs?????????(
? "%?"
?
0?????????$
? ~
)__inference_output_2_layer_call_fn_108423Q??/?,
%?"
 ?
inputs?????????(
? "??????????$?
D__inference_output_3_layer_call_and_return_conditional_losses_108454^??/?,
%?"
 ?
inputs?????????(
? "%?"
?
0?????????$
? ~
)__inference_output_3_layer_call_fn_108443Q??/?,
%?"
 ?
inputs?????????(
? "??????????$?
D__inference_output_4_layer_call_and_return_conditional_losses_108474^??/?,
%?"
 ?
inputs?????????(
? "%?"
?
0?????????$
? ~
)__inference_output_4_layer_call_fn_108463Q??/?,
%?"
 ?
inputs?????????(
? "??????????$?
$__inference_signature_wrapper_107345?:,-<=>?FGVWXY`apqrs????????????????????A?>
? 
7?4
2
images(?%
images?????????dd"W?T
R
tf.compat.v1.transpose_16?3
tf.compat.v1.transpose_1?????????$