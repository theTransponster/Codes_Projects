import java.awt.BorderLayout;

import javax.swing.JFrame;

import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.videoio.VideoCapture;

public class VentanaHipoFlexion extends JFrame {

	private PanelPrincipal pp;
	private PanelHipoFlexion p;
	private PanelControles p1;
	public int indicador;
	
	public VentanaHipoFlexion(){
		super("Tratamiento HipoFlexion");
		this.setDefaultCloseOperation(EXIT_ON_CLOSE);
		this.setLocation(100, 50);
		this.p=new PanelHipoFlexion(this);
		this.p1=new PanelControles(p);
		p.escribir(p1);
		this.add(p);
		this.add(p1,BorderLayout.EAST);
		
		this.pack();
		
		this.setVisible(true);
	}
	public void setDatos(PanelPrincipal pp){
		this.pp=pp;
		this.p.setDatos(this.pp.tNombre.getText(), this.pp.tEdad.getText(), this.pp.tFecha.getText(), this.pp.tTiempo.getText());
		if(this.pp.auxiliar==1){
			this.indicador=1;
			this.p.setCoords(300, 150, 300, 450, 500, 350, 80, 50, 90, 220);
		}else{
			this.p.setCoords(280, 150, 80, 50, 20, 380, 130, 200, 285, 380);
			this.indicador=2;
		}
	}

}
