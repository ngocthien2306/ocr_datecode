declare module 'react-plotly.js' {
  import { Component } from 'react';
  import { PlotParams } from 'plotly.js';

  export interface PlotProps extends Partial<PlotParams> {
    data: Partial<PlotParams['data']>;
    layout?: Partial<PlotParams['layout']>;
    config?: Partial<PlotParams['config']>;
    frames?: Partial<PlotParams['frames']>;
    style?: React.CSSProperties;
    className?: string;
    onInitialized?: (figure: Readonly<PlotParams>, graphDiv: Readonly<HTMLElement>) => void;
    onUpdate?: (figure: Readonly<PlotParams>, graphDiv: Readonly<HTMLElement>) => void;
    onPurge?: (figure: Readonly<PlotParams>, graphDiv: Readonly<HTMLElement>) => void;
    onError?: (err: Readonly<Error>) => void;
    onClick?: (event: Readonly<any>) => void;
    onHover?: (event: Readonly<any>) => void;
    onUnhover?: (event: Readonly<any>) => void;
    onSelected?: (event: Readonly<any>) => void;
    revision?: number;
    divId?: string;
    useResizeHandler?: boolean;
    debug?: boolean;
  }

  export default class Plot extends Component<PlotProps> {}
}
