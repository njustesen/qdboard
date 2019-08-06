'use strict';

var app = angular.module('app', ['ngRoute', 'ngSanitize', 'appControllers', 'appServices', 'appDirectives', 'appFilters']);

var appServices = angular.module('appServices', []);
var appControllers = angular.module('appControllers', []);
var appDirectives = angular.module('appDirectives', []);
var appFilters = angular.module('appFilters', []);

var options = {};
options.api = {};
options.api.base_url = "http://127.0.0.1:5000";


app.config(['$locationProvider', '$routeProvider', 
  function($location, $routeProvider) {
    $routeProvider.
        when('/', {
            templateUrl: 'static/partials/run.list.html',
            controller: 'RunListCtrl'
        }).
        when('/run/create', {
            templateUrl: 'static/partials/run.create.html',
            controller: 'RunListCtrl',
            access: { requiredAuthentication: true }
        }).
        when('/run/show/:id', {
            templateUrl: 'static/partials/run.show.html',
            controller: 'RunListCtrl',
            access: { requiredAuthentication: true }
        }).
        otherwise({
            redirectTo: '/'
        });
}]);

