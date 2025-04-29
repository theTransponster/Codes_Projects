import java.awt.Color;
import java.awt.Dimension;
import java.awt.Graphics;
import java.awt.Graphics2D;
import java.awt.Image;
import java.awt.RenderingHints;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.awt.image.BufferedImage;
import java.awt.image.DataBufferByte;
import java.io.ByteArrayInputStream;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

import javax.imageio.ImageIO;
import javax.swing.ImageIcon;
import javax.swing.JButton;
import javax.swing.JFrame;
import javax.swing.JPanel;

import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfByte;
import org.opencv.core.MatOfPoint;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.highgui.HighGui;
import org.opencv.highgui.*;
//import org.opencv.videoio.VideoCapture;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import org.opencv.imgproc.*;
import org.opencv.videoio.VideoCapture;

public class PanelHipoFlexion extends JPanel implements Runnable{
	private JButton btIniciar,
					btExpediente,
					btParar;
	private static final long serialVersionUID = 1L;
	private BufferedImage imagen; 
    public VentanaHipoFlexion vh;
    public boolean aux=true;
    private VideoCapture webCam;
    private int x1,
    			x2,
    			x3,
    			x4,
    			x5,
    			y1,
    			y2,
    			y3,
    			y4,
    			y5,
    			satMax,
    			satMin,
    			valMax,
    			valMin,
    			matMax,
    			matMin,
    			pos;
    private Mat hsv_image,
    			thresholded,
    			thresholded2,
    			nueva,
    			im,
    			imagenDeWebCam;
	private int posx,
				posy;
	private Image fondo;
	private PanelControles pc;
	private String estado;
	private String[] datos;
	private int[] coordenadas;

	public PanelHipoFlexion(VentanaHipoFlexion vh){
		super();
		this.pos=1;
		this.x1=300; this.y1=350;
		this.x2=150; this.y2=80;
		this.x3=300; this.y3=50;
		this.x4=450; this.y4=90;
		this.x5=500; this.y5=220;
		this.matMin=90; this.matMax=100;
		this.satMin=135; this.satMax=200;
		this.valMin=159; this.valMax=255;
		this.coordenadas=new int[11];
		this.estado="Esperando movimiento...";
		this.datos=new String[4];
		System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
		MatOfByte mb2 = new MatOfByte();
		this.hsv_image=new Mat();
		this.thresholded=new Mat();
		this.thresholded2=new Mat();
		imagenDeWebCam = new Mat();
		webCam = new VideoCapture(0);
		Thread hilo=new Thread(this);
		hilo.start();
		Mat m=Mat.ones(400, 400,CvType.CV_8UC1);
		Imgcodecs.imencode("ima.jpg", m, mb2);
		try {
			this.imagen=ImageIO.read(new ByteArrayInputStream(mb2.toArray()));
		} catch (IOException e3) {
			e3.printStackTrace();
		}
		this.setBackground(Color.white);
		this.vh=vh;
		this.setPreferredSize(new Dimension(600,520));
		this.btIniciar=new JButton("Iniciar prueba");
		this.btIniciar.addActionListener(new ActionListener(){
			public void actionPerformed(ActionEvent e){
						System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
			}
		});
		this.btParar=new JButton("Detener");
		this.btParar.addActionListener(new ActionListener(){
			public void actionPerformed(ActionEvent e){
				webCam.release();
				aux=false;
			}
		});
		this.add(this.btParar);
	}
	
	public boolean convierteMatABufferedImage(Mat matriz) {
		Imgproc.blur(matriz, thresholded, new Size(7,7));
		Imgproc.cvtColor(thresholded,hsv_image,Imgproc.COLOR_RGB2BGR);
		Imgproc.cvtColor(hsv_image, hsv_image, Imgproc.COLOR_BGR2HSV);
		Scalar lower = new Scalar(this.matMin, this.satMin, this.valMin);
	    Scalar upper = new Scalar(this.matMax, this.satMax, this.valMax);
	    Mat hsvFrame = new Mat(matriz.rows(), matriz.cols(), CvType.CV_8U, new Scalar(3));
        Imgproc.cvtColor(matriz, hsvFrame, Imgproc.COLOR_RGB2HSV, 3);
        Mat skinMask = new Mat(hsvFrame.rows(), hsvFrame.cols(), CvType.CV_8U, new Scalar(3));
        Core.inRange(hsvFrame, lower, upper, skinMask);
		MatOfByte mb = new MatOfByte();
		Mat dilateElement = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, new Size(20, 20));
		Mat erodeElement = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, new Size(4, 4));
		Imgproc.erode(skinMask, skinMask, erodeElement);
		Imgproc.dilate(skinMask, skinMask, dilateElement);
		Imgproc.dilate(skinMask, skinMask, dilateElement);
		List<MatOfPoint> contours = new ArrayList<>();
		Mat hierarchy = new Mat();

		 Imgproc.findContours(skinMask, contours, hierarchy, Imgproc.RETR_CCOMP, Imgproc.CHAIN_APPROX_SIMPLE);
		 
		 hierarchy.get(posy, posx);
		 if (hierarchy.size().height > 0 && hierarchy.size().width > 0)
		 {		getCoords();
		         for (int idx = 0; idx >= 0; idx = (int) hierarchy.get(0, idx)[0])
		         {
		                 Imgproc.drawContours(matriz, contours, idx, new Scalar(250, 0, 0));
		                 Rect rect = Imgproc.boundingRect(contours.get(idx));
		                 if((((double)rect.x)<=(coordenadas[2]*1.20)&&((double)rect.x)>=(coordenadas[2]-(coordenadas[2]*.20)))
		                		 &&(((double)rect.y)<=(coordenadas[3]*1.30)&&((double)rect.y)>=(coordenadas[3]-(coordenadas[3]*.30))) ){
		                	 this.setEstado("Posición de inicio");
		                	 pos=2;
		                 }  
		                 if((((double)rect.x)<=(coordenadas[4]*1.20)&&((double)rect.x)>=(coordenadas[4]-(coordenadas[4]*.20)))
		                		 &&(((double)rect.y)<=(coordenadas[5]*1.30)&&((double)rect.y)>=(coordenadas[5]-(coordenadas[5]*.30))) && pos==2){
		                	if(this.vh.indicador==1){	 
		                		this.setEstado("Flexión de 90°");
		                	}else{
		                		this.setEstado("Extensión de 120°");
		                	}
		                	 pos=3;
		                 }  
		                 if((((double)rect.x)<=(coordenadas[6]*1.20)&&((double)rect.x)>=(coordenadas[6]-(coordenadas[6]*.20)))
		                		 &&(((double)rect.y)<=(coordenadas[7]*1.30)&&((double)rect.y)>=(coordenadas[7]-(coordenadas[7]*.30))) && pos==3){
		                	 if(this.vh.indicador==1){
		                		 this.setEstado("Flexión de 60°");
		                	 }else{
		                		 this.setEstado("Extensión de 160°");
		                	 }
		                	 pos=4;
		                 } 
		                 if((((double)rect.x)<=(coordenadas[8]*1.20)&&((double)rect.x)>=(coordenadas[8]-(coordenadas[8]*.20)))
		                		 &&(((double)rect.y)<=(coordenadas[9]*1.30)&&((double)rect.y)>=(coordenadas[9]-(coordenadas[9]*.30))) && pos==4){
		                	if(this.vh.indicador==1){
		                		 this.setEstado("Flexión de 45°");
		                	 }else{
		                		 this.setEstado("Extensión de 180°");
		                	 }
		                 }  
		         }
		 }
		Imgcodecs.imencode("ima.jpg", matriz, mb);
		try {
			this.imagen = ImageIO.read(new ByteArrayInputStream(mb.toArray()));
			repaint();
		} catch (IOException e) {
			e.printStackTrace();
			return false; 
		}
		return true; 
	}	
	public void paintComponent(Graphics g) {
		super.paintComponent(g);
		this.getCoords();
		BufferedImage bufImage = new BufferedImage(this.imagen.getWidth(), this.imagen.getHeight(), this.imagen.getType());
		if (this.imagen == null){
			return;
		}
		else{
			g.drawImage(this.imagen, 0, 0, this.imagen.getWidth(), this.imagen.getHeight(), null);
			g.setColor(Color.blue);
			g.drawOval(this.coordenadas[0], this.coordenadas[1], 50, 50);
			g.setColor(Color.green);
			g.drawOval(this.coordenadas[2], this.coordenadas[3], 50, 50);
			g.setColor(Color.RED);
			g.drawOval(this.coordenadas[4], this.coordenadas[5], 50, 50);
			g.drawOval(this.coordenadas[6], this.coordenadas[7], 50, 50);
			g.drawOval(this.coordenadas[8], this.coordenadas[9], 50, 50);
		}
	}
	public void run() {
		System.out.println(this.aux);
		try{
			while(this.aux!=false){
				webCam.read(imagenDeWebCam);
				this.getMatMin(); this.getMatMax(); this.getSatMax(); this.getSatMin();
				this.getValMax(); this.getValMin();
				convierteMatABufferedImage(imagenDeWebCam);
				repaint();
				Thread.sleep(50);
			}	
		}catch(InterruptedException e){
			System.out.println("No se pudo ejecutar");
			}
		}
	
	public int setSatMax(int satMax){
		return this.satMax=satMax;
	}
	public int setSatMin(int satMin){
		return this.satMin=satMin;
	}
	public int setValMax(int valMax){
		return this.valMax=valMax;
	}
	public int setValmin(int valMin){
		return this.valMin=valMin;
	}
	public int setMatMax(int matMax){
		return this.matMax=matMax;
	}
	public int setMatMin(int matMin){
		return this.matMin=matMin;
	}
	public int getMatMin(){
		return this.matMin;
	}
	public int getMatMax(){
		return this.matMax;
	}
	public int getSatMin(){
		return this.satMin;
	}
	public int getSatMax(){
		return this.satMax;
	}
	public int getValMin(){
		return this.valMin;
	}
	public int getValMax(){
		return this.valMax;
	}
	public void setEstado(String es){
		this.estado=es;
		this.pc.setMov(this.estado);
		this.pc.listaMovs(this.estado);
	}
	public String getEstado(){
		return this.estado;
	}
	public void escribir(PanelControles pc){
		this.pc=pc;
		this.pc.setMov(this.estado);
	}
	public void setCoords(int x1, int x2, int x3, int x4,int x5, int y1, int y2, int y3, int y4, int y5){
		this.coordenadas[0]=x1; this.coordenadas[1]=y1; this.coordenadas[2]=x2; this.coordenadas[3]=y2;
		this.coordenadas[4]=x3; this.coordenadas[5]=y3; this.coordenadas[6]=x4; this.coordenadas[7]=y4;
		this.coordenadas[8]=x5; this.coordenadas[9]=y5;
		getCoords();
	}
	public int[] getCoords(){
		return this.coordenadas;
	}
	public void setDatos(String Nombre, String Edad, String Fecha,String Tiempo){
		this.datos[0]=Nombre;
		this.datos[1]=Edad;
		this.datos[2]=Tiempo;
		this.datos[3]=Fecha;
	}	
	public String[] getDatos(){
		return this.datos;
	}

}

   	

   
	 


