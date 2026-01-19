import api from './http';
import { authAPI } from './auth';
import { usersAPI } from './users';
import recipesAPI, { receiptsAPI } from './recipes';
import { camerasAPI } from './cameras';
import { uploadAPI } from './upload';
import { actionLogsAPI } from './actionLogs';

export { authAPI, usersAPI, recipesAPI, receiptsAPI, camerasAPI, uploadAPI, actionLogsAPI };
export default api;
