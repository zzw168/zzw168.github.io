
// 设置画布初始属性
const canvasMain = document.querySelector('.canvasMain');
const canvas = document.getElementById('canvas');
const resultGroup = document.querySelector('.resultGroup');

// 设置画布宽高背景色
canvas.width = canvas.clientWidth;
canvas.height = canvas.clientHeight;
canvas.style.background = "#8c919c";

const annotate = new LabelImage({
	canvas: canvas,
	scaleCanvas: document.querySelector('.scaleCanvas'),
	scalePanel: document.querySelector('.scalePanel'),
	annotateState: document.querySelector('.annotateState'),
	canvasMain: canvasMain,
	resultGroup: resultGroup,
	crossLine: document.querySelector('.crossLine'),
	labelShower: document.querySelector('.labelShower'),
	screenShot: document.querySelector('.screenShot'),
	screenFull: document.querySelector('.screenFull'),
	colorHex: document.querySelector('#colorHex'),
	toolTagsManager: document.querySelector('.toolTagsManager'),
	historyGroup: document.querySelector('.historyGroup')
});

// 初始化交互操作节点
const prevBtn = document.querySelector('.pagePrev');                    // 上一张
const nextBtn = document.querySelector('.pageNext');                    // 下一张
const taskName = document.querySelector('.pageName');                   // 标注任务名称
const processIndex = document.querySelector('.processIndex');           // 当前标注进度
const processSum = document.querySelector('.processSum');               // 当前标注任务总数

let imgFiles = [ './images/example/football.jpg', './images/example/person.jpg', './images/example/band.jpg',
'./images/example/street.jpg', './images/example/dog.jpeg', './images/example/cat.jpg', './images/example/dogs.jpg',
'./images/example/furniture.jpg', './images/ex0ample/basketball.jpg', './images/example/alley.jpg'];    //选择上传的文件数据集
//let imgFiles = [];
let imgIndex = 1;       //标定图片默认下标;
let imgSum = 10;        // 选择图片总数;

initImage();
// 初始化图片状态
function initImage() {
	selectImage(0);
	processSum.innerText = imgSum;
}

//切换操作选项卡
let tool = document.getElementById('tools');
tool.addEventListener('click', function(e) {
	for (let i=0; i<tool.children.length; i++) {
		tool.children[i].classList.remove('focus');
	}
	e.target.classList.add('focus');
	switch(true) {
		case e.target.className.indexOf('toolDrag') > -1:  // 拖拽
			annotate.SetFeatures('dragOn', true);
			break;
		case e.target.className.indexOf('toolRect') > -1:  // 矩形
			annotate.SetFeatures('rectOn', true);
			break;
		case e.target.className.indexOf('toolPolygon') > -1:  // 多边形
			annotate.SetFeatures('polygonOn', true);
			break;
		case e.target.className.indexOf('toolTagsManager') > -1:  // 标签管理工具
			annotate.SetFeatures('tagsOn', true);
			break;
		default:
			break;
	}
});

// 获取下一张图片
nextBtn.onclick = function() {
	annotate.Arrays.imageAnnotateMemory.length > 0 && localStorage.setItem(taskName.textContent, JSON.stringify(annotate.Arrays.imageAnnotateMemory));  // 保存已标定的图片信息
	if (imgIndex >= imgSum) {
		imgIndex = 1;
		selectImage(0);
	}
	else {
		imgIndex++;
		selectImage(imgIndex - 1);
	}
};

// 获取上一张图片
prevBtn.onclick = function() {
	annotate.Arrays.imageAnnotateMemory.length > 0 && localStorage.setItem(taskName.textContent, JSON.stringify(annotate.Arrays.imageAnnotateMemory));  // 保存已标定的图片信息
	if (imgIndex === 1) {
		imgIndex = imgSum;
		selectImage(imgSum - 1);
	}
	else {
		imgIndex--;
		selectImage(imgIndex - 1);
	}
};

document.querySelector('.openFolder').addEventListener('click', function() {
	document.querySelector('.openFolderInput').click()
});

function changeFolder(e) {
	imgFiles = e.files;
	imgSum = imgFiles.length;
	processSum.innerText = imgSum;
	imgIndex = 1;
	selectImage(0);
}

function selectImage(index) {
	openBox('#loading', true);
	processIndex.innerText = imgIndex;
	taskName.innerText = imgFiles[index].name || imgFiles[index].split('/')[3];
	let content = localStorage.getItem(taskName.textContent);
	let img = imgFiles[index].name ? window.URL.createObjectURL(imgFiles[index]) : imgFiles[index];
	content ? annotate.SetImage(img, JSON.parse(content)) : annotate.SetImage(img);
}

document.querySelector('.saveJson').addEventListener('click', function() {
	let filename = taskName.textContent.split('.')[0] + '.json';
	annotate.Arrays.imageAnnotateMemory.length > 0 ? saveJson(annotate.Arrays.imageAnnotateMemory, filename): alert('当前图片未有有效的标定数据');
});

function saveJson(data, filename) {
	if (!data) {
		alert('保存的数据为空');
		return false;
	}
	if (!filename) {
		filename = 'json.json';
	}
	if (typeof data === 'object') {
		data = JSON.stringify(data, undefined, 4);
	}
	let blob = new Blob([data], {type: 'text/json'}),
		e = document.createEvent('MouseEvent'),
		a = document.createElement('a');
		a.download = filename;
		a.href = window.URL.createObjectURL(blob);
		a.dataset.downloadurl = ['text/json', a.download, a.href].join(':');
		e.initMouseEvent('click', true, false, window,  0, 0, 0, 0, 0, false, false, false, false, 0, null);
		a.dispatchEvent(e)
}

//弹出框
function openBox (e, isOpen) {
	let el = document.querySelector(e);
	let maskBox = document.querySelector('.mask_box');
	if (isOpen) {
		maskBox.style.display = "block";
		el.style.display = "block";
	}
	else {
		maskBox.style.display = "none";
		el.style.display = "none";
	}
}