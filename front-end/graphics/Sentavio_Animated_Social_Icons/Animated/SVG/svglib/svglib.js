
(function(exportType) {
  var SVGlib = {
    utils: {
      loadDoc: function(url, callback) {
        var xhttp = new XMLHttpRequest();
          xhttp.onreadystatechange = function() {
            if (xhttp.readyState == 4 && xhttp.status == 200) {
              callback.call(null, xhttp);
            }
          };
          xhttp.open("GET", url, true);
          xhttp.send();
      },
      ready: function(fn) {
        if (document.readyState != 'loading'){
          fn();
        } else {
          document.addEventListener('DOMContentLoaded', fn);
        }
      }
    },
    animationConfig: {
      interval: null,
      looped: true,
      scene: null,
      frameTime: 77,
      delay: 0,
      onOver: false,
      reverse: false,
      direction: 'forward'
    },
    currentFrame: 1,
    animationStarted: false,
    step: 1,
    startFrame: 1,
    endFrame: 13,
    firstOut: false,
    reversed: false,

    format: function(num) {
      if(num < 10) {
        return '0' + num;
      }
      return num.toString();
    },

    encodeForInsertion: function(string) {
      return "data:image/svg+xml," + encodeURIComponent(string);
    },

    nextFrame: function () {
      if(this.currentFrame === this.endFrame) {

        if(!this.animationConfig.looped) this.stopAnimation();

        this.currentFrame = this.startFrame;
      }

      this.animationConfig.scene.setAttribute('xlink:href', '#frame' + this.format(this.currentFrame));
      this.currentFrame += this.step;

      if(this.animationConfig.image) {
        this.animationConfig.image.setAttribute('src', this.encodeForInsertion(this.animationConfig.root.outerHTML));
      }
    },

    startAnimation: function(ev) {
      var self = this;
      setTimeout(function() {
        self.animationStarted = true;
        self.interval = setInterval(function() {
          self.nextFrame();
        },self.animationConfig.frameTime);
      }, this.animationConfig.delay);
    },

    stopAnimation: function(ev) {
      clearInterval(this.interval);
      this.animationStarted = false;
      if(this.reversed) this.reverseAnimation();

    },

    isInside: function(mouseEvent, clientRect) {
      return (
        (mouseEvent.clientX >= clientRect.left) && (mouseEvent.clientX <= clientRect.right)
        &&
        (mouseEvent.clientY >= clientRect.top) && (mouseEvent.clientY <= clientRect.bottom)
      );
    },

    reverseAnimation: function() {
        this.startFrame = this.startFrame === 1 ? this.endFrame : 1;
        this.endFrame = this.endFrame !== 1 ? 1 : this.animationConfig.frameCount;
        this.step = this.step === -1 ? 1 : -1;
        if(!this.animationStarted) {
            this.currentFrame = this.startFrame;
        }
        this.reversed = !this.reversed;
    },

    handleMouseMove: function(ev, isInside) {

      var self = this;

      if(typeof isInside === 'undefined') isInside = self.isInside(ev, self.animationConfig.scene.getBoundingClientRect());

      if(isInside && !self.animationStarted && !self.firstOut) {
        self.startAnimation();
        self.firstOut = true;
      } else if(!isInside && self.animationStarted && self.animationConfig.reverse && self.firstOut) {
        self.reverseAnimation();
      } else if(!isInside && self.animationStarted && self.firstOut) {
        self.stopAnimation();
      }

      if(!isInside && self.firstOut) {
        self.firstOut = false;
      }
    },

    init: function(config) {
      for(var setting in config){
        if(config.hasOwnProperty(setting)) {
          this.animationConfig[setting] = config[setting]
        }
      }

      if(config.frameCount) this.endFrame = config.frameCount;

      if(this.animationConfig.direction !== 'forward') {
        this.reverseAnimation();
		this.reversed = false;
      }

      var self = this;

      if(this.animationConfig.onOver && this.animationConfig.image) {
        this.animationConfig.image.addEventListener('mouseenter', function(ev){
          self.handleMouseMove(ev, true);
        });

        this.animationConfig.image.addEventListener('mouseleave', function(ev){
          self.handleMouseMove(ev, false);
        });

      } else if (this.animationConfig.onOver) {
        window.addEventListener('mousemove', function(ev){
          self.handleMouseMove(ev);
        });
      } else {
        self.startAnimation();
      }
    },

    initEmbedded: function() {
      var elem = document.querySelector('.animation');
          this.init({
            scene: elem.querySelector('.animation__scene use'),
            delay: parseInt(elem.getAttribute('data-delay'), 10),
            frameTime: parseInt(elem.getAttribute('data-frame'), 10),
            looped: (elem.getAttribute('data-looped') === 'true'),
            onOver: (elem.getAttribute('data-onover') === 'true'),
            reverse: (elem.getAttribute('data-reverse') === 'true'),
            direction: elem.getAttribute('data-direction'),
            frameCount: elem.querySelectorAll('symbol').length
          });
    },

    initAsImg: function(xml) {
      var i,
          xmlDoc = xml.responseXML,
          elem = xmlDoc.querySelector('.animation');

      this.init({
        root: elem,
        scene: elem.querySelector('.animation__scene use'),
        delay: parseInt(elem.getAttribute('data-delay'), 10),
        frameTime: parseInt(elem.getAttribute('data-frame'), 10),
        looped: (elem.getAttribute('data-looped') === 'true'),
        onOver: (elem.getAttribute('data-onover') === 'true'),
        reverse: (elem.getAttribute('data-reverse') === 'true'),
        direction: elem.getAttribute('data-direction'),
        image:  document.querySelector('#svg__holder'),
        frameCount: elem.querySelectorAll('symbol').length
      });
    }
  };

  var utils = SVGlib.utils;

  utils.ready(function(){
    if(exportType === 'embedded') {
      SVGlib.initEmbedded();
    } else if(exportType === 'image') {
      var image = document.querySelector('#svg__holder');
      utils.loadDoc(image.getAttribute('src'), function(resp){
        SVGlib.initAsImg(resp);
      });
    }
  });
})((window.state && window.state.exportType) || 'embedded');
